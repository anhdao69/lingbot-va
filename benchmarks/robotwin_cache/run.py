"""Speed/memory only: official RoboTwin policy on fixed released camera images.

No simulator, success metric, teacher target, optimizer or training loop.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.gdn.common import (
    EventProfiler,
    cache_bytes,
    cuda_timed,
    load_model,
    parser,
)
from benchmarks.robotwin_cache.analyze import validate_configuration
from benchmarks.robotwin_cache.state import restore_policy, snapshot_policy

PROMPT = "Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack."


def compact(profile):
    sums = defaultdict(lambda: {"ms": 0.0, "count": 0})
    for event in profile.pop("events"):
        item = sums[(event["module"], event["phase"], event["update_cache"])]
        item["ms"] += event["ms"]
        item["count"] += 1
    profile["events"] = [
        dict(module=k[0], phase=k[1], update_cache=k[2], **v) for k, v in sums.items()
    ]
    return profile


def main():
    p = parser()
    p.set_defaults(config="robotwin", dense_backend="dense_fast")
    p.add_argument(
        "--history-chunks", type=int, nargs="+", default=[4, 8, 15, 22, 30, 36]
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--profile-repetitions", type=int, default=2)
    p.add_argument("--images", default=str(ROOT / "example/robotwin"))
    a = p.parse_args()
    if a.config != "robotwin" or a.adapter:
        p.error("This phase uses unadapted RoboTwin only")
    if a.variant not in ("dense", "dense_fast", "local100", "gdn50", "gdn75", "gdn100"):
        p.error("Choose one of the six requested configurations")
    if a.variant == "dense":
        a.dense_backend = "full"
    if a.repetitions < 2 or a.profile_repetitions < 2 or a.warmup < 1:
        p.error("Use warmup and at least two clean/profile repetitions")
    validate_configuration(vars(a), a.variant)
    m = load_model(a)
    out = Path(a.out)
    paths = [Path(a.images) / (k + ".png") for k in m.job_config.obs_cam_keys]
    image = {
        k: np.array(Image.open(f).convert("RGB"))
        for k, f in zip(m.job_config.obs_cam_keys, paths)
    }
    provenance = json.loads((out / "config.json").read_text())
    provenance["image_sha256"] = {
        str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f): hashlib.sha256(
            f.read_bytes()
        ).hexdigest()
        for f in paths
    }
    provenance["robotwin_benchmark_sha256"] = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (ROOT / "benchmarks/robotwin_cache").glob("*.py")
    }
    receipt = (
        Path(a.checkpoint)
        / ".cache/huggingface/download/transformer/config.json.metadata"
    )
    downloaded_revision = (
        receipt.read_text().splitlines()[0] if receipt.exists() else None
    )
    if downloaded_revision != "8c9dea8abbc5c91cc9e18bc3264b8915083bbe70":
        raise ValueError(
            "Download the pinned official RoboTwin checkpoint; revision receipt is missing or different"
        )
    provenance["checkpoint_revision"] = downloaded_revision
    provenance["checkpoint_config_sha256"] = hashlib.sha256(
        (Path(a.checkpoint) / "transformer/config.json").read_bytes()
    ).hexdigest()
    provenance.update(
        expected_checkpoint_revision="8c9dea8abbc5c91cc9e18bc3264b8915083bbe70",
        prompt=PROMPT,
        observation_protocol="fixed released camera images; zero observed actions; CPU snapshots outside timing",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        attention_layers=len(m.transformer.blocks),
        batch=2,
        heads=24,
        head_dim=128,
        video_tokens=240,
        action_tokens=32,
        cache_capacity=9792,
        video_model_calls=26,
        action_model_calls=51,
    )
    (out / "config.json").write_text(json.dumps(provenance, indent=2))
    m.infer({"reset": True, "prompt": PROMPT})
    # Exactly the first observation encoding used by the deployed server.
    with torch.no_grad():
        m.init_latent = m._encode_obs({"obs": [image]})
    zero_action = np.zeros(
        (len(m.job_config.used_action_channel_ids), 2, 16), dtype=np.float32
    )
    profiler = EventProfiler(m)
    prefix = 0
    rows = []
    targets = sorted(set(a.history_chunks))
    for target in targets:
        if not 1 <= target <= 40:
            raise ValueError("History chunks must be within 1..40")
        while prefix < target:
            m.infer(
                {
                    "obs": [image] * (4 if prefix == 0 else 8),
                    "compute_kv_cache": True,
                    "state": zero_action,
                }
            )
            prefix += 1
        saved = snapshot_policy(m)
        for phase, repetitions in [
            ("warmup", a.warmup),
            ("clean", a.repetitions),
            ("profile", a.profile_repetitions),
        ]:
            for repetition in range(repetitions):
                restore_policy(m, saved)
                torch.manual_seed(a.seed + repetition)
                torch.cuda.manual_seed_all(a.seed + repetition)
                torch.cuda.reset_peak_memory_stats()
                before = cache_bytes(m.transformer)
                if phase == "profile":
                    profiler.start()
                reply, policy_ms, policy_wall_ms = cuda_timed(
                    m.infer, {"obs": image, "prompt": PROMPT}
                )
                profile = compact(profiler.stop()) if phase == "profile" else None
                generation_memory = cache_bytes(m.transformer)
                if not np.isfinite(reply["action"]).all():
                    raise RuntimeError("Non-finite generated action")
                if phase == "profile":
                    calls = profile["model_calls"]
                    assert sum(c["phase"] == "video" for c in calls) == 26
                    assert sum(c["phase"] == "action" for c in calls) == 51
                    assert {
                        c["query_tokens"] for c in calls if c["phase"] == "video"
                    } == {240}
                    assert {
                        c["query_tokens"] for c in calls if c["phase"] == "action"
                    } == {32}
                    profiler.start()
                _, commit_ms, commit_wall_ms = cuda_timed(
                    m.infer,
                    {
                        "obs": [image] * 8,
                        "compute_kv_cache": True,
                        "state": zero_action,
                    },
                )
                commit_profile = (
                    compact(profiler.stop()) if phase == "profile" else None
                )
                row = {
                    "variant": a.variant,
                    "history_chunks": target,
                    "nominal_history_tokens": min(272 * target, 9792),
                    "phase": phase,
                    "repetition": repetition,
                    "policy_ms": policy_ms,
                    "policy_wall_ms": policy_wall_ms,
                    "commit_ms": commit_ms,
                    "commit_wall_ms": commit_wall_ms,
                    "cycle_ms": policy_ms + commit_ms,
                    "historical_bytes_before": before,
                    "historical_bytes_peak": max(
                        before, generation_memory, cache_bytes(m.transformer)
                    ),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "profile": profile,
                    "commit_profile": commit_profile,
                }
                rows.append(row)
                (out / "measurements.json").write_text(json.dumps(rows, indent=2))
                print(
                    "MEASURED",
                    a.variant,
                    target,
                    phase,
                    repetition,
                    round(policy_ms, 2),
                    round(commit_ms, 2),
                    flush=True,
                )
        restore_policy(m, saved)
        del saved
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
