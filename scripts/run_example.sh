#!/usr/bin/env bash
# Reproduce the growth-model prediction for one subject/hemisphere with the
# published parameters (sigma_R = 1.30, sigma_T = 2.20, alpha = 0.30).
#
# Usage:
#   bash scripts/run_example.sh            # default: R1 lh
#   bash scripts/run_example.sh S1 rh      # any subject (R1, S1..S6) and hemisphere (lh/rh)
set -euo pipefail

DATA="${1:-R1_gpr_grid}"
TAG="${2:-lh}"

# Resolve repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Point the loader at the pkls bundled in this repo.
export SHARED_DATA_ROOT="$REPO_ROOT/data"

# Published hyperparameters.
SIGMA_R=1.30      # radial kernel width  (--radius)
SIGMA_T=2.20      # tangential kernel width in degrees (--tangent)
ALPHA=0.30        # resource-decay weight (--alpha)

cd "$REPO_ROOT/src"
python experiment.py \
    --data "$DATA" \
    --tag "$TAG" \
    --mode mds \
    --distance_mode polar \
    --algo deterministic \
    --radius "$SIGMA_R" \
    --tangent "$SIGMA_T" \
    --alpha "$ALPHA"

SUFFIX="$(printf '%.2f_%.2f_%.2f' "$SIGMA_R" "$SIGMA_T" "$ALPHA")"
echo
echo "Outputs (relative to repo root):"
echo "  outputs/predictions/mds/predicted_${DATA}_${TAG}_deterministic_${SUFFIX}.tsv"
echo "  outputs/plots/${DATA}_${TAG}_tuning_compare_${SUFFIX}.png"
