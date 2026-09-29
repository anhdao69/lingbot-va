#!/usr/bin/env bash
set -euo pipefail
# Run from repository root after activating the documented environment.
# Existing evaluation processes are identified by --port and retained.
: "${GDN_RECORDING:?Set GDN_RECORDING to the directory containing obs_data_*.pt}"
GDN_GPU=${GDN_GPU:-0}
GDN_PORT=${GDN_PORT:-30056}
GDN_SUFFIX=${GDN_SUFFIX:-_final}
for variant in dense gdn100 local100 gdn50 local50 gdn75 local75; do
  python benchmarks/gdn/isolated.py --gpu "$GDN_GPU" --port "$GDN_PORT" --timeout 900 \
    --log "docs/gdn_data/replay_${variant}${GDN_SUFFIX}.log" -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=1 benchmarks/gdn/replay.py \
    --variant "$variant" --chunks 18 --warmup 3 --instrument --replicated-gates \
    --recording "$GDN_RECORDING" --out "docs/gdn_data/replay_${variant}${GDN_SUFFIX}"
done
