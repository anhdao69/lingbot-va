"""Generate measured comparison tables and publication-style static plots."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["dense", "local50", "local75", "local100", "gdn50", "gdn75", "gdn100"]
LABELS = [
    "Dense",
    "Local 50%",
    "Local 75%",
    "Local 100%",
    "GDN 50%",
    "GDN 75%",
    "GDN 100%",
]


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="docs/gdn_data")
    p.add_argument("--figures", default="docs/gdn_figures")
    p.add_argument("--suffix", default="_final")
    a = p.parse_args()
    data = Path(a.data)
    figures = Path(a.figures)
    figures.mkdir(parents=True, exist_ok=True)
    summaries = {}
    replays = {}
    for variant in VARIANTS:
        d = data / ("replay_" + variant + a.suffix)
        summaries[variant] = read(d / "summary.json")
        replays[variant] = read(d / "replay.compact.json")
    dense = summaries["dense"]
    gib = 1024**3
    mib = 1024**2
    lines = [
        "| Model | Self-attn ms | Block ms | Video ms | Action ms | Clean policy ms | Peak allocated GiB | Attn speedup | Policy speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, label in zip(VARIANTS, LABELS):
        s = summaries[variant]
        lines.append(
            f"| {label} | {s['self_attn_ms']:.1f} | {s['block_ms']:.1f} | {s['video_ms']:.1f} | {s['action_ms']:.1f} | {s['policy_ms']:.1f} | {s['peak_allocated_bytes'] / gib:.3f} | {dense['self_attn_ms'] / s['self_attn_ms']:.3f}× | {dense['policy_ms'] / s['policy_ms']:.3f}× |"
        )
    lines += [
        "",
        "Self-attention and block columns sum all 30 layers across all 72 denoising calls. Video/action spans include intervening scheduler/host dispatch between Transformer calls. These four columns use instrumented calls; clean policy times use separate calls without profiler hooks. Their totals need not equal clean policy latency. All times are milliseconds.",
        "",
        "| Model | Persistent cache peak MiB | Cache saving | Observed cache-update ms | Policy calls/s | Action-equivalent Hz (with commit) | Full cycle ms | Full cycle speedup | Peak reserved GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, label in zip(VARIANTS, LABELS):
        s = summaries[variant]
        lines.append(
            f"| {label} | {s['max_cache_bytes'] / mib:.1f} | {100 * (1 - s['max_cache_bytes'] / dense['max_cache_bytes']):.1f}% | {s['cache_update_ms']:.1f} | {1000 / s['policy_ms']:.3f} | {16000 / (s['policy_ms'] + s['cache_update_ms']):.3f} | {s['policy_ms'] + s['cache_update_ms']:.1f} | {(dense['policy_ms'] + dense['cache_update_ms']) / (s['policy_ms'] + s['cache_update_ms']):.3f}× | {s['peak_reserved_bytes'] / gib:.3f} |"
        )
    lines += [
        "",
        "Persistent GDN cache peak includes committed **and** speculative states. Allocator peak also includes transient state-update/output tensors. Hz excludes simulator, transport, model load and reset. The policy emits 16 actions per steady-state call; action-equivalent Hz is not a sensor feedback rate.",
        "",
        "| Model | Clean samples | Clean policy mean ± SD ms | Instrumented policy mean ms | Cross-attn ms | FFN ms | Mean one-block invocation ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, label in zip(VARIANTS, LABELS):
        rows = replays[variant]
        clean = [
            r["policy_gpu_ms"] for r in rows if not r["warmup"] and not r["profiled"]
        ]
        prof = [r["policy_gpu_ms"] for r in rows if r["profiled"]]
        s = summaries[variant]
        lines.append(
            f"| {label} | {len(clean)} | {np.mean(clean):.1f} ± {np.std(clean, ddof=1):.1f} | {np.mean(prof):.1f} | {s['cross_attn_ms']:.1f} | {s['ffn_ms']:.1f} | {s['block_ms'] / 2160:.3f} |"
        )
    synth = read(data / "synthetic_final" / "synthetic.json")
    lines += [
        "",
        "### Synthetic eager attention (ms)",
        "",
        "| Query tokens | Historical tokens | Dense SDPA | Dense gather + SDPA | Local | Local + GDN read | Local + GDN read/update |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in (16, 128):
        for h in (1000, 2000, 5000, 10000, 20000):
            r = {
                x["backend"]: x["ms"]
                for x in synth["rows"]
                if x["query_tokens"] == t and x["history_tokens"] == h
            }
            lines.append(
                f"| {t} | {h} | "
                + " | ".join(
                    f"{r[k]:.4f}"
                    for k in [
                        "dense_core",
                        "dense_gather_core",
                        "local",
                        "gdn_read_local",
                        "gdn_commit_local",
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "These CUDA-event intervals include eager Python launch gaps, as deployment does. They are not CUDA-graph-only hardware throughput. History construction is excluded from repeated reads; commit measurements include gate materialization and FLA recurrence.",
        "",
        "| Query tokens | Update/read kernel | ms |",
        "|---:|---|---:|",
    ]
    for r in synth["updates"]:
        lines.append(f"| {r['query_tokens']} | {r['kernel']} | {r['ms']:.4f} |")
    lines += [
        "",
        "### Actual dense LIBERO history",
        "",
        "| Chunk | Video history before | Action history before | Profiled policy ms | Self-attn ms | Cross-attn ms | FFN ms | Block ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in read(data / "dense_libero_profile" / "calls.compact.json"):
        if r["kind"] != "policy" or not r.get("profile"):
            continue
        prof = r["profile"]
        calls = prof["model_calls"]
        sums = {
            k: sum(x["ms"] for x in prof["events"] if x["module"] == k)
            for k in ("self_attn", "cross_attn", "ffn", "block")
        }
        histories = [
            next(c["history_tokens_before"] for c in calls if c["phase"] == phase)
            for phase in ("video", "action")
        ]
        lines.append(
            f"| {r['chunk']} | {histories[0]} | {histories[1]} | {r['gpu_ms']:.1f} | "
            + " | ".join(f"{sums[k]:.1f}" for k in sums)
            + " |"
        )
    lines += [
        "",
        "Chunk 0 is cold and excluded from steady-state conclusions. This first actual rollout used an earlier, heavier profiler (including nested SDPA events); its absolute times are not the clean comparative timings above.",
        "",
        "### LIBERO smoke checks",
        "",
        "| Model | Adaptation | Task | Initial state | Success | Episode seconds |",
        "|---|---|---:|---:|---|---:|",
    ]
    for d in sorted(data.glob("*")):
        f = d / "smoke_results.json"
        if not f.exists():
            continue
        for r in read(f):
            if (
                d.name == "smoke_local100_before"
                and r["task"] == 5
                and (data / "smoke_local100_matched/smoke_results.json").exists()
            ):
                continue  # Replaced by the run with per-episode matched RNG.
            lines.append(
                f"| {d.name} | {'after' if 'after' in d.name else 'none'} | {r['task']} | {r['episode']} | {r['success']} | {r['seconds']:.1f} |"
            )
    alignment = read(data / "alignment" / "alignment.json")
    lines += [
        "",
        "### Teacher-forced attention alignment",
        "",
        "| Split | Local NMSE | GDN before NMSE | GDN after NMSE |",
        "|---|---:|---:|---:|",
    ]
    for heldout in (False, True):
        rs = [r for r in alignment["samples"] if r["heldout"] == heldout]
        lines.append(
            "| "
            + ("held-out chunk 14" if heldout else "train chunks 1/5/10")
            + " | "
            + " | ".join(
                f"{np.mean([r[k] for r in rs]):.5f}"
                for k in ("local_nmse", "gdn_before_nmse", "gdn_after_nmse")
            )
            + " |"
        )
    lines += [
        "",
        "| Held-out phase | Local NMSE | GDN before NMSE | GDN after NMSE |",
        "|---|---:|---:|---:|",
    ]
    for phase in ("video", "action"):
        rs = [r for r in alignment["samples"] if r["heldout"] and r["phase"] == phase]
        lines.append(
            "| "
            + phase
            + " | "
            + " | ".join(
                f"{np.mean([r[k] for r in rs]):.5f}"
                for k in ("local_nmse", "gdn_before_nmse", "gdn_after_nmse")
            )
            + " |"
        )
    lines += [
        "",
        f"Alignment: {alignment['steps_per_layer']} Adam steps/layer, learning rate {alignment['lr']}, {alignment['parameter_count']} trainable parameters, {alignment['elapsed_s']:.1f} s including setup/evaluation. No original checkpoint weights were loaded into the optimizer process.",
    ]
    (data / "measured_results.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update(
        {"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, t in zip(axes, (16, 128)):
        for backend, label in [
            ("dense_core", "Dense SDPA"),
            ("dense_gather_core", "Dense gather + SDPA"),
            ("local", "Local"),
            ("gdn_read_local", "Local + GDN read"),
            ("gdn_commit_local", "Local + GDN commit"),
        ]:
            rs = sorted(
                [
                    r
                    for r in synth["rows"]
                    if r["query_tokens"] == t and r["history_tokens"] >= 1000
                ],
                key=lambda r: r["history_tokens"],
            )
            rs = [r for r in rs if r["backend"] == backend]
            ax.plot(
                [r["history_tokens"] for r in rs],
                [r["ms"] for r in rs],
                marker=".",
                label=label,
            )
        ax.set(
            xlabel="Historical tokens",
            ylabel="Eager CUDA-event latency (ms)",
            title=f"Q={t}, B=2, H=24, D=128",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "synthetic_latency.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for variant, label in zip(VARIANTS, LABELS):
        rs = [r for r in replays[variant] if not r["warmup"] and not r["profiled"]]
        axes[0].plot(
            [min(r["chunk"] * 144, 2160) for r in rs],
            [r["policy_gpu_ms"] for r in rs],
            marker=".",
            label=label,
        )
        axes[1].plot(
            [min(r["chunk"] * 144, 2160) for r in rs],
            [r["peak_allocated_bytes"] / gib for r in rs],
            marker=".",
            label=label,
        )
    axes[0].set(
        xlabel="Nominal dense history tokens", ylabel="Clean policy latency (ms)"
    )
    axes[1].set(
        xlabel="Nominal dense history tokens",
        ylabel="Peak allocated GPU memory (GiB)",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "policy_history.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for backend, label in [
        ("dense_core", "Dense explicit history"),
        ("gdn_read_local", "GDN one state"),
        ("local", "Local"),
    ]:
        rs = sorted(
            [
                r
                for r in synth["rows"]
                if r["query_tokens"] == 128 and r["backend"] == backend
            ],
            key=lambda r: r["history_tokens"],
        )
        ax.plot(
            [r["history_tokens"] for r in rs],
            [r["persistent_history_bytes"] / mib for r in rs],
            marker=".",
            label=label,
        )
    ax.set(xlabel="Historical tokens", ylabel="One-layer persistent history (MiB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "history_memory.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, phase in zip(axes, ("video", "action")):
        rs = sorted(
            [r for r in alignment["samples"] if r["heldout"] and r["phase"] == phase],
            key=lambda r: r["layer"],
        )
        for key, label in (
            ("local_nmse", "Local"),
            ("gdn_before_nmse", "GDN before"),
            ("gdn_after_nmse", "GDN after"),
        ):
            ax.plot(
                [r["layer"] for r in rs], [r[key] for r in rs], marker=".", label=label
            )
        ax.set(
            xlabel="Layer", ylabel="Normalized attention MSE", title=f"Held-out {phase}"
        )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "alignment.png")
    plt.close(fig)
    print(data / "measured_results.md")


if __name__ == "__main__":
    main()
