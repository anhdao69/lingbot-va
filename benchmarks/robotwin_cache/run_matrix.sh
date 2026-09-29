#!/usr/bin/env bash
set -euo pipefail
GPU=${GDN_GPU:-0}
PORT=${GDN_PORT:-30056}
ROOT=${ROBOTWIN_OUT:-docs/dense_fast_robotwin_data}
KERNEL=${GDN_KERNEL:-auto}
# Use a single GPU for the matrix. The original owned evaluation lane is safely
# paused/resumed by the existing watchdog helper for each bounded command.
for variant in dense_fast dense gdn100 local100 gdn50 gdn75; do
  python benchmarks/gdn/isolated.py --gpu "$GPU" --port "$PORT" --timeout 2400 \
    --log "$ROOT/${variant}.log" -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=1 benchmarks/robotwin_cache/run.py \
    --variant "$variant" --kernel "$KERNEL" --out "$ROOT/$variant"
done
python benchmarks/gdn/isolated.py --gpu "$GPU" --port "$PORT" --timeout 2400 \
  --log "$ROOT/dense_fast_repeat.log" -- \
  torchrun --standalone --nnodes=1 --nproc_per_node=1 benchmarks/robotwin_cache/run.py \
  --variant dense_fast --kernel "$KERNEL" --out "$ROOT/dense_fast_repeat"
python benchmarks/robotwin_cache/analyze.py --data "$ROOT"
