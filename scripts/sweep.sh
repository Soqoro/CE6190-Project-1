#!/usr/bin/env bash
set -euo pipefail

# Simple sweep launcher without external tools (no yq needed).
# Usage:
#   ./scripts/sweep.sh                 # defaults: ds=voc, model=deeplab, seeds 0 1 2, pcts 1 5 10 100
#   DS=voc MODEL=deeplab ./scripts/sweep.sh
#   DS=kvasir MODEL=unet SEEDS="0" PCTS="5 10" ./scripts/sweep.sh
#
# Environment variables:
#   DS     : dataset (voc | kvasir | parts), default "voc"
#   MODEL  : model family (unet | deeplab | segformer), default "deeplab"
#   SEEDS  : space-separated seeds, default "0 1 2"
#   PCTS   : space-separated label percentages, default "1 5 10 100"

DS="${DS:-voc}"
MODEL="${MODEL:-deeplab}"
SEEDS="${SEEDS:-0 1 2}"
PCTS="${PCTS:-1 5 10 100}"

echo "Sweep: ds=${DS}, model=${MODEL}, seeds=[${SEEDS}], pcts=[${PCTS}]"

for seed in $SEEDS; do
  for pct in $PCTS; do
    echo ">> Running ds=${DS}, model=${MODEL}, seed=${seed}, pct=${pct}"
    python -m scripts.launch --ds "${DS}" --model "${MODEL}" --seed "${seed}" --pct "${pct}"
  done
done
