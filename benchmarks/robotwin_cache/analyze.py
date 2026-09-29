"""Generate measured RoboTwin tables; never substitute synthetic or missing runs."""

import argparse
import json
import statistics
from pathlib import Path

VARIANTS = ["dense", "dense_fast", "local100", "gdn50", "gdn75", "gdn100"]
LABELS = {
    "dense": "Original dense",
    "dense_fast": "Dense fast",
    "local100": "Local 100%",
    "gdn50": "GDN 50% + dense_fast",
    "gdn75": "GDN 75% + dense_fast",
    "gdn100": "GDN 100%",
}


def validate_configuration(config, variant):
    if config.get("variant") != variant or config.get("config") != "robotwin":
        raise ValueError("Variant/config does not match the RoboTwin comparison")
    expected = "full" if variant == "dense" else "dense_fast"
    if config.get("dense_backend") != expected:
        raise ValueError(f"{variant} requires dense_backend={expected}")
    if config.get("adapter"):
        raise ValueError("This comparison must use unadapted GDN parameters")


def speedup_over_dense_fast(summaries, variant):
    return summaries["dense_fast"]["policy_ms"] / summaries[variant]["policy_ms"]


def decision(long_history_speedups):
    conservative = min(long_history_speedups)
    if conservative >= 1.25:
        return "GO"
    if conservative >= 1.10:
        return "CONDITIONAL"
    return "STOP speed direction"


def summarize(rows):
    clean = [r for r in rows if r["phase"] == "clean"]
    profiled = [r for r in rows if r["phase"] == "profile"]
    if len(clean) < 2 or len(profiled) < 2:
        raise ValueError("Incomplete run: need repeated clean and profile samples")
    result = {
        k: statistics.mean(r[k] for r in clean)
        for k in ["policy_ms", "commit_ms", "cycle_ms"]
    }
    result["policy_sd_ms"] = statistics.stdev(r["policy_ms"] for r in clean)
    result["clean_samples"] = len(clean)
    for k in ["peak_allocated_bytes", "peak_reserved_bytes", "historical_bytes_peak"]:
        result[k] = max(r[k] for r in clean)
    for key in ["self_attn", "cross_attn", "ffn", "block", "transformer"]:
        result[key + "_ms"] = statistics.mean(
            sum(e["ms"] for e in r["profile"]["events"] if e["module"] == key)
            for r in profiled
        )
    for key in ["video", "action"]:
        result[key + "_ms"] = statistics.mean(
            r["profile"]["phase_spans"][key + "_span_ms"] for r in profiled
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="docs/dense_fast_robotwin_data")
    p.add_argument("--figures", default="docs/dense_fast_robotwin_figures")
    a = p.parse_args()
    data, figures = Path(a.data), Path(a.figures)
    for variant in VARIANTS:
        config = json.loads((data / variant / "config.json").read_text())
        validate_configuration(config, variant)
    all_rows = {
        v: json.loads((data / v / "measurements.json").read_text()) for v in VARIANTS
    }
    histories = sorted({r["nominal_history_tokens"] for r in all_rows["dense_fast"]})
    if len(histories) != 6 or histories[-2:] != [8160, 9792]:
        raise ValueError("Decision report requires all six requested history points")
    summaries = {
        h: {
            v: summarize([r for r in rows if r["nominal_history_tokens"] == h])
            for v, rows in all_rows.items()
        }
        for h in histories
    }
    (data / "summary.json").write_text(json.dumps(summaries, indent=2))
    lines = [
        "# Measured RoboTwin speed/memory results",
        "",
        "Latency units: ms. Self-attention, blocks, Transformer, video and action use separate instrumented repetitions. Policy/commit/cycle use clean repetitions without hooks. Cycle = policy + observed-cache commit, including VAE encoding. CPU snapshot transfers and reset/prompt/model loading are outside all timings.",
        "",
        "| History | Model | Self-attn | Block | Transformer | Video | Action | Policy ± SD | Commit | Peak GiB | History MiB | Policy speedup vs dense_fast |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in histories:
        for v in VARIANTS:
            s = summaries[h][v]
            cells = [
                f"{s[k + '_ms']:.2f}"
                for k in ["self_attn", "block", "transformer", "video", "action"]
            ]
            lines.append(
                f"| {h} | {LABELS[v]} | "
                + " | ".join(cells)
                + f" | {s['policy_ms']:.2f} ± {s['policy_sd_ms']:.2f} | {s['commit_ms']:.2f} | {s['peak_allocated_bytes'] / 1024**3:.3f} | {s['historical_bytes_peak'] / 1024**2:.1f} | {speedup_over_dense_fast(summaries[h], v):.3f}× |"
            )
    lines += [
        "",
        "| History | Model | Self-attn speedup over dense_fast | Cycle ms | Cycle speedup over dense_fast |",
        "|---:|---|---:|---:|---:|",
    ]
    for h in histories:
        for v in VARIANTS:
            s, b = summaries[h][v], summaries[h]["dense_fast"]
            lines.append(
                f"| {h} | {LABELS[v]} | {b['self_attn_ms'] / s['self_attn_ms']:.3f}× | {s['cycle_ms']:.2f} | {b['cycle_ms'] / s['cycle_ms']:.3f}× |"
            )
    lines += [
        "",
        "## Decision by GDN fraction",
        "",
        "Apply the threshold conservatively to the smaller measured policy speedup at 8160 and 9792 nominal tokens. This is a screening rule, not a statistical confidence statement. CONDITIONAL requires considering the measured memory benefit below.",
        "",
    ]
    for v in ["gdn50", "gdn75", "gdn100"]:
        rates = [speedup_over_dense_fast(summaries[h], v) for h in histories[-2:]]
        saving = (
            1
            - summaries[histories[-1]][v]["historical_bytes_peak"]
            / summaries[histories[-1]]["dense_fast"]["historical_bytes_peak"]
        )
        lines.append(
            f"- {LABELS[v]}: **{decision(rates)}**, {rates[0]:.3f}× / {rates[1]:.3f}× policy speedups; {saving:.1%} less persistent GPU history at 9792 tokens."
        )
    repeat_path = data / "dense_fast_repeat/measurements.json"
    if repeat_path.exists():
        repeated = json.loads(repeat_path.read_text())
        lines += [
            "",
            "## Dense-fast repeat drift",
            "",
            "| History | Initial policy ms | Repeat policy ms | Drift |",
            "|---:|---:|---:|---:|",
        ]
        for h in histories:
            s = summarize([r for r in repeated if r["nominal_history_tokens"] == h])
            base = summaries[h]["dense_fast"]["policy_ms"]
            lines.append(
                f"| {h} | {base:.2f} | {s['policy_ms']:.2f} | {s['policy_ms'] / base - 1:+.1%} |"
            )
    else:
        raise ValueError(
            "Repeat dense_fast baseline missing; do not finalize the comparison"
        )
    (data / "measured_results.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, key, label in zip(
        axes.flat,
        ["policy_ms", "self_attn_ms", "transformer_ms", "cycle_ms"],
        [
            "Clean policy decision",
            "Self-attention module sum",
            "Transformer forward sum",
            "Policy + observed commit",
        ],
    ):
        for v in VARIANTS:
            ax.plot(
                histories,
                [summaries[h][v][key] for h in histories],
                marker=".",
                label=LABELS[v],
            )
        ax.set(
            xlabel="Nominal dense history tokens", ylabel="Latency (ms)", title=label
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "latency.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, key, scale, label in zip(
        axes,
        ["peak_allocated_bytes", "historical_bytes_peak"],
        [1024**3, 1024**2],
        ["Peak GPU allocation (GiB)", "Persistent KV/state (MiB)"],
    ):
        for v in VARIANTS:
            ax.plot(
                histories,
                [summaries[h][v][key] / scale for h in histories],
                marker=".",
                label=LABELS[v],
            )
        ax.set(xlabel="Nominal dense history tokens", ylabel=label)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "memory.png", dpi=160)
    plt.close(fig)
    print(data / "measured_results.md")


if __name__ == "__main__":
    main()
