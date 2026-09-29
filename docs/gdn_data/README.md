# GDN measured artifacts

The report is [../gdn_benchmark_report.md](../gdn_benchmark_report.md).

- `replay_*_final/`: the final seven-configuration matrix. `config.json` records arguments and measured source hashes, `replay.compact.json` preserves per-call timing and aggregated module events, and `summary.json` holds clean/instrumented means separately.
- `replay_dense_repeat/`: independent dense repeat and a separate operator trace, taken after all timed samples.
- `synthetic_final/`: CUDA-event synthetic kernel measurements.
- `alignment/`: gate-only fit metrics and the small 2,160-parameter adapter; no base-model weights.
- `smoke_*_before/`, `smoke_*_after/`: actual LIBERO rollouts, with one initial state each for tasks 0 and 5. These do not estimate final benchmark success rate. `smoke_local100_matched/` supersedes the earlier task-5 Local-100 result for matched RNG comparison.
- `dense_libero_profile/`: original actual-environment profile and Day-1 reproduction. This predates the lighter final profiling hooks.
- `gates_shard/`, `gates_replicated/`, `gate_equivalence.json`: FSDP gate overhead and exact full-model output comparison.
- Unsuffixed `replay_*`, `gdn_probe/`, `synthetic/`, and `exploratory_results.md`: exploratory measurements, retained for provenance. They are not the final matrix and should not be mixed with it.

Large individual-event JSONL traces, observation/teacher captures, videos, downloaded datasets, checkpoints and environments remain on the H100 workspace. Operational pause/watchdog files are excluded. The compact records preserve numerical measurement results without publishing these assets. Source hashes reflect the code at measurement time; later report/analysis edits can change their hashes without changing the inference implementation.
