# Dense-fast RoboTwin Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; preserve the user's existing `gdn` branch.

**Goal:** Exact dense cache optimization and speed/memory-only RoboTwin comparison, with GDN speedups relative to dense_fast.

**Architecture:** Keep Q/K/V, norms, RoPE, SDPA and output projection intact. Store active K/V contiguously, track legacy virtual slots/ages/speculation on CPU, append temporary tokens to spare suffix space, and compact only on eviction or prediction clearing. Preserve the original unstable argsort tie selection with a bounded shared metadata-plan cache; cold ambiguous eviction plans may use the original device sort, rather than silently changing token survival. Original dense remains selectable. Hybrid unconverted layers use dense_fast only in the new benchmark.

**Tech Stack:** Existing PyTorch 2.9, CUDA, FLA/Triton, pytest; official RoboTwin checkpoint and three example camera images. No simulator/SR or training.

**Spec:** User's 2026-09-29 request; preserve original cache behavior, including irreversible eviction by temporary calls.

**Review Focus:**
- Temporary overflow keeps its eviction after rollback; ordinary temporary calls do not change history.
- Equal-age partial evictions retain exactly the original tokens on CPU and CUDA.
- Predicted video conditions actions; clear_pred drops predictions without resurrecting evicted observations.
- Independent cache names and reset cannot leak data across episodes.
- Denominator, history positions, warmup/repetitions and allocation accounting stay comparable across variants.

## Tasks

- [x] Add failing numerical/state tests against original dense for normal, speculative, temporary, partial-eviction and reset sequences.
- [x] Implement `wan_va/modules/dense_cache.py`, dispatch from `model.py`, add configurable dense fallback and cache memory accounting; pass CPU/CUDA tests.
- [x] Pin/download official RoboTwin checkpoint, verify 30-layer / actual query/cache shapes and GPU availability.
- [x] Add speed-only RoboTwin harness using original server inference and frozen example observations, approximately 1/2/4/6/8/10k histories, multiple clean repetitions and separate module profiling. Include cache update timing and actual peak/persistent memory.
- [x] Run all six configurations with identical inputs and seeds; run repeated dense_fast controls. Save compact metrics and provenance.
- [x] Generate tables/plots and `docs/dense_fast_robotwin_benchmark.md`; decide GO/CONDITIONAL/STOP using GDN versus dense_fast. Review, verify, preserve original evaluations, commit/push gdn under existing authorization.
