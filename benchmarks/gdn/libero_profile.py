"""Actual LIBERO rollout with sparse CUDA-event instrumentation, or smoke eval."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.gdn.common import (
    EventProfiler,
    cache_bytes,
    cuda_timed,
    load_model,
    parser,
)
from evaluation.libero import client


def main():
    p = parser()
    p.add_argument("--tasks", nargs="+", type=int, default=[0, 5])
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    m = load_model(a)
    initial_cpu_rng = torch.get_rng_state()
    initial_cuda_rng = torch.cuda.get_rng_state_all()
    out = Path(a.out)
    (out / "calls.jsonl").write_text("")
    profiler = EventProfiler(m) if a.profile else None
    original_construct = client.construct_single_env

    def construct(args):
        env = original_construct(args)
        if env is None:
            raise RuntimeError("Failed to create LIBERO environment")
        env.seed(a.seed)
        return env

    client.construct_single_env = construct

    class Policy:
        chunk = 0
        reset_start = 0

        def infer(self, obs):
            if obs.get("reset"):
                self.chunk = 0
                return m.infer(obs)
            is_cache = obs.get("compute_kv_cache", False)
            selected = a.profile and self.chunk in (0, 3, 8, 15, 24)
            if selected:
                profiler.start()
            result, gpu_ms, wall_ms = cuda_timed(m.infer, obs)
            profile = profiler.stop() if selected else None
            row = {
                "task": current_task,
                "episode": current_episode,
                "chunk": self.chunk,
                "kind": "cache" if is_cache else "policy",
                "gpu_ms": gpu_ms,
                "wall_ms": wall_ms,
                "profiled": selected,
                "cache_bytes": cache_bytes(m.transformer),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            if profile:
                row["profile"] = profile
            with (out / "calls.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if is_cache:
                self.chunk += 1
            return result

    policy = Policy()
    results = []
    torch.cuda.reset_peak_memory_stats()
    for current_task in a.tasks:
        for current_episode in range(a.episodes):
            # Keep identical denoising noise across tasks regardless of when the
            # preceding variant terminated its episode.
            torch.set_rng_state(initial_cpu_rng)
            torch.cuda.set_rng_state_all(initial_cuda_rng)
            begin = time.perf_counter()
            success = client.run_one(
                policy, "libero_10", current_task, str(out / "videos"), current_episode
            )
            row = {
                "task": current_task,
                "episode": current_episode,
                "success": bool(success),
                "rng_policy": "restore_post_model_load_each_episode",
                "seconds": time.perf_counter() - begin,
            }
            results.append(row)
            (out / "smoke_results.json").write_text(json.dumps(results, indent=2))
            print("EPISODE_RESULT", json.dumps(row), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
