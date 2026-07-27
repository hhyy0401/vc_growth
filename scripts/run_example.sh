#!/usr/bin/env bash
# Reproduce the growth-model prediction for one subject/hemisphere with the
# published parameters (sigma_R = 1.30, sigma_T = 2.20, alpha = 0.30).
#
# Usage:
#   bash scripts/run_example.sh
#   bash scripts/run_example.sh --data S1_gpr_grid --tag rh
#   bash scripts/run_example.sh --data R1_gpr_grid --sigma-r 1.30 --sigma-t 2.20
#
# Backward-compatible positional form:
#   bash scripts/run_example.sh S1_gpr_grid rh
set -euo pipefail

usage() {
    sed -n '2,12p' "$0"
}

DATA="R1_gpr_grid"
TAG="lh"
SIGMA_R="1.30"
SIGMA_T="2.20"

if [[ $# -gt 0 && "$1" != --* ]]; then
    DATA="$1"
    TAG="${2:-lh}"
    shift "$(( $# >= 2 ? 2 : 1 ))"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            DATA="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --sigma-r)
            SIGMA_R="$2"
            shift 2
            ;;
        --sigma-t)
            SIGMA_T="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# Resolve repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Point the loader at the pkls bundled in this repo.
export SHARED_DATA_ROOT="$REPO_ROOT/data"

# Remaining published defaults are intentionally fixed here.
ALPHA=0.30

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
