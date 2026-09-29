"""Identical recorded-observation streams for timing each attention variant."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gdn.common import (
    EventProfiler,
    cache_bytes,
    cuda_timed,
    load_model,
    parser,
)


def main():
    p = parser()
    p.add_argument("--recording", required=True)
    p.add_argument("--chunks", type=int, default=18)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--instrument", action="store_true")
    p.add_argument("--operators", action="store_true")
    a = p.parse_args()
    m = load_model(a)
    out = Path(a.out)
    (out / "replay.jsonl").write_text("")
    files = sorted(
        Path(a.recording).glob("obs_data_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    obs = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
    assert len(obs) > 1 and len(obs[0]) == 12 and all(len(x) == 16 for x in obs[1:])
    prompt = "put both the alphabet soup and the tomato sauce in the basket"
    m.infer({"reset": True, "prompt": prompt})
    prof = EventProfiler(m) if a.instrument else None
    rows = []
    for i in range(a.chunks):
        warm = i < a.warmup
        if not warm:
            torch.cuda.reset_peak_memory_stats()
        selected = bool(prof and i in (3, 8, 15, 17))
        if selected:
            prof.start()
        reply, gpu_ms, wall_ms = cuda_timed(
            m.infer, {"obs": obs[0][0], "prompt": prompt}
        )
        detail = prof.stop() if selected else None
        generation_cache_bytes = cache_bytes(m.transformer)
        assert np.isfinite(reply["action"]).all()
        frame_obs = obs[0] if i == 0 else obs[1 + (i - 1) % (len(obs) - 1)]
        if selected:
            prof.start()
        _, cache_ms, cache_wall = cuda_timed(
            m.infer,
            {
                "obs": frame_obs,
                "compute_kv_cache": True,
                "imagine": False,
                "state": reply["action"],
            },
        )
        cache_detail = prof.stop() if selected else None
        row = {
            "chunk": i,
            "warmup": warm,
            "profiled": selected,
            "generation_cache_bytes": generation_cache_bytes,
            "policy_gpu_ms": gpu_ms,
            "policy_wall_ms": wall_ms,
            "cache_gpu_ms": cache_ms,
            "cache_wall_ms": cache_wall,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cache_bytes": cache_bytes(m.transformer),
            "profile": detail,
            "cache_profile": cache_detail,
        }
        rows.append(row)
        with (out / "replay.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            "REPLAY",
            a.variant,
            i,
            round(wall_ms, 2),
            round(cache_wall, 2),
            row["cache_bytes"],
            flush=True,
        )
    steady = [r for r in rows if not r["warmup"] and not r["profiled"]]
    result = {
        "variant": a.variant,
        "adapter": a.adapter,
        "chunks": len(steady),
        "policy_ms": float(np.mean([r["policy_gpu_ms"] for r in steady])),
        "policy_wall_ms": float(np.mean([r["policy_wall_ms"] for r in steady])),
        "cache_update_ms": float(np.mean([r["cache_gpu_ms"] for r in steady])),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in steady),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in steady),
        "max_cache_bytes": max(r["generation_cache_bytes"] for r in steady),
    }
    if prof:
        sums = {
            key: []
            for key in [
                "self_attn",
                "cross_attn",
                "ffn",
                "block",
                "dense_core",
                "video",
                "action",
            ]
        }
        for r in rows:
            if r["profile"] is None:
                continue
            e = r["profile"]["events"]
            for key in ["self_attn", "cross_attn", "ffn", "block", "dense_core"]:
                sums[key].append(sum(x["ms"] for x in e if x["module"] == key))
            for phase in ["video", "action"]:
                sums[phase].append(r["profile"]["phase_spans"][phase + "_span_ms"])
        result.update({k + "_ms": float(np.mean(v)) for k, v in sums.items()})
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print("SUMMARY", json.dumps(result), flush=True)
    if a.operators:
        # Separate diagnostic pass; excluded from all reported clean timings.
        originals = []
        for block in m.transformer.blocks:
            for label, attention in (
                ("self_sdpa", block.attn1),
                ("cross_sdpa", block.attn2),
            ):
                original = attention.attn_op
                originals.append((attention, original))

                def scoped(q, k, v, original=original, label=label):
                    with torch.profiler.record_function("gdn." + label):
                        return original(q, k, v)

                attention.attn_op = scoped
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as operator_profile:
            m.infer({"obs": obs[0][0], "prompt": prompt})
            torch.cuda.synchronize()
        for attention, original in originals:
            attention.attn_op = original
        operators = []
        for e in operator_profile.key_averages():
            operators.append(
                {
                    "name": e.key,
                    "device_type": str(e.device_type),
                    "count": e.count,
                    "self_cpu_ms": e.self_cpu_time_total / 1000,
                    "self_device_ms": e.self_device_time_total / 1000,
                    "total_device_ms": e.device_time_total / 1000,
                    "total_cpu_ms": e.cpu_time_total / 1000,
                }
            )
        kernels = [
            e for e in operator_profile.events() if str(e.device_type).endswith("CUDA")
        ]
        (out / "operators.json").write_text(
            json.dumps(
                {
                    "device_event_ms": sum(e.device_time_total for e in kernels) / 1000,
                    "device_event_count": len(kernels),
                    "operators": operators,
                },
                indent=2,
            )
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
