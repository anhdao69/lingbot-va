"""CUDA-event microbenchmarks at LingBot's actual B/H/D and query sizes."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wan_va.modules.history_attention import GDNHistory, local_softmax


def timed(fn, warmup=10, repeats=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Initialize events before measurement, and reuse them after synchronization.
    start.record()
    end.record()
    end.synchronize()
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    values.sort()
    return {
        "ms": sum(values) / len(values),
        "median_ms": values[len(values) // 2],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def gathered(q, k, v, mask):
    valid = mask.nonzero(as_tuple=False).squeeze(-1)
    return local_softmax(q, k[:, valid], v[:, valid])


def hybrid(mem, q, k, v, state, commit=False):
    output = mem.read(q, state, local_softmax(q, k, v))
    if commit:
        return output, mem.update_state(k, v, state)
    return output


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--queries", nargs="+", type=int, default=[16, 128])
    parser.add_argument(
        "--histories",
        nargs="+",
        type=int,
        default=[0, 144, 576, 1152, 2016, 1000, 2000, 5000, 10000, 20000],
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rows, updates = [], []
    for tokens in args.queries:
        q, k, v = [
            torch.randn(2, tokens, 24, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        state = torch.randn(2, 24, 128, 128, device="cuda") * 0.1
        mem = GDNHistory(24, 128, read_kernel="triton").cuda().requires_grad_(False)
        for kernel in ("chunk", "recurrent"):
            mem.kernel = kernel
            row = dict(
                query_tokens=tokens,
                kernel=kernel,
                **timed(partial(mem.update_state, k, v, state)),
            )
            updates.append(row)
            print("UPDATE", row, flush=True)
        mem.kernel = "auto"
        local = local_softmax(q, k, v)
        for kernel in ("torch", "triton"):
            mem.read_kernel = kernel
            row = dict(
                query_tokens=tokens,
                kernel="read_" + kernel,
                **timed(partial(mem.read, q, state, local)),
            )
            updates.append(row)
            print("READ", row, flush=True)
        mem.read_kernel = "triton"
        for history in args.histories:
            kh = torch.randn(
                2, history + tokens, 24, 128, device="cuda", dtype=torch.bfloat16
            )
            vh = torch.randn_like(kh)
            mask = torch.ones(history + tokens, device="cuda", dtype=torch.bool)
            functions = {
                "dense_core": partial(local_softmax, q, kh, vh),
                "dense_gather_core": partial(gathered, q, kh, vh, mask),
                "local": partial(local_softmax, q, k, v),
                "gdn_read_local": partial(hybrid, mem, q, k, v, state),
                "gdn_commit_local": partial(hybrid, mem, q, k, v, state, commit=True),
            }
            for name, fn in functions.items():
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                row = dict(
                    query_tokens=tokens,
                    history_tokens=history,
                    backend=name,
                    **timed(fn),
                    incremental_peak_bytes=torch.cuda.max_memory_allocated() - base,
                    persistent_history_bytes=0
                    if name == "local"
                    else (
                        state.numel() * 4
                        if name.startswith("gdn")
                        else history * 2 * 24 * 128 * 2 * 2
                    ),
                )
                rows.append(row)
            print("HISTORY", tokens, history, flush=True)
            del functions, fn, kh, vh, mask
    data = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "batch": 2,
        "heads": 24,
        "head_dim": 128,
        "warmup": 10,
        "repeats": 40,
        "updates": updates,
        "rows": rows,
    }
    (out / "synthetic.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
