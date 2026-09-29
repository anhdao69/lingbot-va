import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wan_va"))
from configs import VA_CONFIGS
from distributed.util import init_distributed
from modules.history_attention import cache_bytes
from wan_va_server import VA_Server, init_logger

VARIANTS = {
    "dense": ("full", 0),
    "dense_fast": ("dense_fast", 1),
    "local50": ("local", 0.5),
    "local75": ("local", 0.75),
    "local100": ("local", 1),
    "gdn50": ("local_gdn", 0.5),
    "gdn75": ("local_gdn", 0.75),
    "gdn100": ("local_gdn", 1),
}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default=None,
    )
    p.add_argument("--variant", choices=VARIANTS, default="dense")
    p.add_argument("--config", choices=["libero", "robotwin"], default="libero")
    p.add_argument("--dense-backend", choices=["full", "dense_fast"], default="full")
    p.add_argument("--adapter")
    gates = p.add_mutually_exclusive_group()
    gates.add_argument(
        "--replicated-gates",
        action="store_true",
        help="Exclude tiny frozen gates from FSDP (default; validated on H100)",
    )
    gates.add_argument(
        "--shard-gates",
        action="store_true",
        help="Control: include tiny gates in FSDP",
    )
    p.add_argument("--kernel", default="auto", choices=["auto", "chunk", "recurrent"])
    p.add_argument("--read-kernel", default="triton", choices=["torch", "triton"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", default=0, type=int)
    return p


def load_model(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    c = VA_CONFIGS[args.config]
    if args.checkpoint is None:
        name = "robotwin" if args.config == "robotwin" else "libero-long"
        args.checkpoint = str(ROOT / ("checkpoints/lingbot-va-posttrain-" + name))
    c.dense_backend = args.dense_backend
    c.wan22_pretrained_model_name_or_path = args.checkpoint
    c.history_backend, c.history_fraction = VARIANTS[args.variant]
    c.gdn_replicated_gates = not args.shard_gates
    c.gdn_kernel = args.kernel
    c.gdn_read_kernel = args.read_kernel
    c.gdn_adapter_path = args.adapter
    c.infer_mode = "server"
    c.save_root = str(Path(args.out) / "debug")
    c.rank = int(os.getenv("RANK", "0"))
    c.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    c.world_size = int(os.getenv("WORLD_SIZE", "1"))
    init_logger()
    init_distributed(c.world_size, c.local_rank, c.rank)
    model = VA_Server(c)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    metadata = dict(vars(args))
    metadata["profiler_version"] = "preallocated_events_v2"
    sources = list((ROOT / "benchmarks/gdn").glob("*.py")) + [
        ROOT / "wan_va/modules/history_attention.py",
        ROOT / "wan_va/modules/dense_cache.py",
        ROOT / "wan_va/modules/history_kernels.py",
        ROOT / "wan_va/modules/model.py",
        ROOT / "wan_va/wan_va_server.py",
        ROOT / "wan_va/distributed/fsdp.py",
        ROOT / "wan_va/distributed/util.py",
        ROOT / "evaluation/libero/client.py",
    ]
    metadata["code_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(sources)
    }
    metadata["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    (Path(args.out) / "config.json").write_text(json.dumps(metadata, indent=2))
    return model


def cuda_timed(fn, *args, **kwargs):
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wall = time.perf_counter()
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end), (time.perf_counter() - wall) * 1000


class EventProfiler:
    """Hooks exist only during measured profiling calls, never clean timing calls."""

    def __init__(self, model, dense_core=False):
        self.model = model
        self.dense_core = dense_core
        self.pool = [torch.cuda.Event(enable_timing=True) for _ in range(24000)]
        for event in self.pool:
            event.record()
        torch.cuda.synchronize()
        self.pool_index = 0
        self.phase = "unknown"
        self.flag = 0
        self.events = []
        self.stacks = {}
        self.handles = []
        self.metadata = []
        self.original_ops = []

    def event(self):
        event = self.pool[self.pool_index]
        self.pool_index += 1
        return event

    def transformer_pre(self, module, args, kwargs):
        self.phase = "action" if kwargs.get("action_mode", False) else "video"
        self.flag = int(kwargs.get("update_cache", 0))
        a = self.model.transformer.blocks[0].attn1
        cache = a.attn_caches.get("pos")
        history = (
            cache.length
            if hasattr(cache, "length")
            else int(cache["mask"].sum().item())
            if cache
            else None
        )
        if a.history_memory is not None:
            snapshot = a.history_memory.snapshot("pos")
            history = snapshot.committed_tokens + snapshot.speculative_tokens
        data = args[0]
        self.metadata.append(
            {
                "phase": self.phase,
                "update_cache": self.flag,
                "history_tokens_before": history,
                "batch": data["noisy_latents"].shape[0],
                "query_tokens": int(np.prod(data["noisy_latents"].shape[2:]))
                // (1 if self.phase == "action" else 4),
            }
        )

    def pre(self, key):
        def hook(module, args):
            a = self.event()
            a.record()
            self.stacks[key] = (a, self.phase, self.flag)

        return hook

    def post(self, key):
        def hook(module, args, result):
            b = self.event()
            b.record()
            a, phase, flag = self.stacks.pop(key)
            self.events.append((*key, phase, flag, a, b))

        return hook

    def start(self):
        assert not self.handles
        self.events = []
        self.metadata = []
        self.pool_index = 0
        transformer = self.model.transformer
        self.handles.append(
            transformer.register_forward_pre_hook(
                self.transformer_pre, with_kwargs=True
            )
        )
        self.handles.append(
            transformer.register_forward_pre_hook(self.pre((-1, "transformer")))
        )
        self.handles.append(
            transformer.register_forward_hook(self.post((-1, "transformer")))
        )
        for i, block in enumerate(transformer.blocks):
            for name, module in [
                ("self_attn", block.attn1),
                ("cross_attn", block.attn2),
                ("ffn", block.ffn),
                ("block", block),
            ]:
                key = (i, name)
                self.handles.append(module.register_forward_pre_hook(self.pre(key)))
                self.handles.append(module.register_forward_hook(self.post(key)))
            if not self.dense_core:
                continue
            original = block.attn1.attn_op
            self.original_ops.append((block.attn1, original))

            def wrapped(q, k, v, original=original, index=i):
                a, b = self.event(), self.event()
                a.record()
                out = original(q, k, v)
                b.record()
                self.events.append((index, "dense_core", self.phase, self.flag, a, b))
                if index == 0:
                    self.metadata[-1]["attended_kv_tokens"] = k.shape[1]
                return out

            block.attn1.attn_op = wrapped

    def stop(self):
        torch.cuda.synchronize()
        for h in self.handles:
            h.remove()
        self.handles = []
        for module, original in self.original_ops:
            module.attn_op = original
        self.original_ops = []
        rows = [
            {
                "layer": i,
                "module": name,
                "phase": phase,
                "update_cache": flag,
                "ms": a.elapsed_time(b),
            }
            for i, name, phase, flag, a, b in self.events
        ]
        bounds = {}
        for phase in ("video", "action"):
            calls = [e for e in self.events if e[1] == "transformer" and e[2] == phase]
            if calls:
                bounds[phase + "_span_ms"] = calls[0][4].elapsed_time(calls[-1][5])
        return {
            "events": rows,
            "phase_spans": bounds,
            "model_calls": self.metadata,
            "cache_bytes": cache_bytes(self.model.transformer),
        }
