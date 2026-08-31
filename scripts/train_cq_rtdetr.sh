#!/usr/bin/env bash
set -Eeuo pipefail

SEED="${SEED:-42}"
CFG="${CFG:-configs/cottonweed/CQ_RTDETR_SR_TCCM.yml}"

python tools/train.py \
    -c "$CFG" \
    --seed "$SEED" \
    --use-amp
