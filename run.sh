#!/bin/bash
cd code/
python experiment.py \
    --data R1_gpr_grid \
    --tag lh \
    --mode mds \
    --distance_mode polar \
    --algo deterministic \
    --radius 2.0 \
    --tangent 2.0
