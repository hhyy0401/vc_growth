# Geometric Constraints in the Development of Primate Extrastriate Visual Cortex

Reference implementation for the macaque retinotopy growth model.
The model simulates feed-forward connectivity growth from V1 into higher visual
areas (V2, V3, V4) on fMRI-derived cortical surface data, then predicts the
retinotopic tuning of the higher areas and compares it against ground truth.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10. A GPU is optional; the model runs on CPU.

## Run

```bash
bash scripts/run_example.sh
bash scripts/run_example.sh --data S1_gpr_grid --tag rh
```

With no arguments it runs R1 LH with sigma_R = 1.30 and sigma_T = 2.20.

Equivalent direct call:

```bash
cd src
SHARED_DATA_ROOT=../data python experiment.py \
    --data R1_gpr_grid --tag lh \
    --mode mds --distance_mode polar --algo deterministic \
    --radius 1.30 --tangent 2.20
```

## Input

`data/{subject}_{hemi}.pkl` is a dict keyed by node ID; each entry holds:

| Field       | Description                                              |
|-------------|----------------------------------------------------------|
| `area`      | Visual area label (1 = V1, 2 = V2, 3 = V3, 4 = V4)       |
| `tuning`    | 2D retinotopic tuning vector `[x, y]` (normalized)       |
| `loc`       | 2D MDS coordinates `[x, y]` used for the kernel geometry |
| `is_center` | `1` for the foveal-center node, `0` otherwise            |

Subjects: `R1` (NMT template) and `S1`–`S6`, which are the six individual
macaques reported as M1–M6 in the paper. Each has a left (`lh`) and right
(`rh`) hemisphere, so 14 files in total.

| Argument      | Meaning                       | Default       |
|---------------|-------------------------------|---------------|
| `--data`      | Subject                       | `R1_gpr_grid` |
| `--tag`       | Hemisphere (`lh` / `rh`)      | `lh`          |
| `--radius`    | Radial kernel width (sigma_R) | `1.30`        |
| `--tangent`   | Tangential kernel width (sigma_T) | `2.20`    |

`run_example.sh` exposes the same four as `--data`, `--tag`, `--sigma-r`,
`--sigma-t`, and keeps `mode=mds`, `distance_mode=polar`,
`algo=deterministic` fixed.

## Output

Written under `outputs/` (git-ignored). For the default run:

| File | Description |
|------|-------------|
| `outputs/predictions/mds/predicted_R1_gpr_grid_lh_deterministic_1.30_2.20.tsv` | Predicted and empirical V2–V4 tuning values |
| `outputs/predictions/mds/W_R1_gpr_grid_lh_deterministic_1.30_2.20.npz` | Model weight matrix and node-generation order |
| `outputs/plots/R1_gpr_grid_lh_tuning_compare_1.30_2.20.png` | Empirical and predicted polar-angle and eccentricity maps |

## Batch-ordering control (Supplementary Fig. S4)

`--custom_batch_mode {angle|polar|euclidean|x}_{fp|pf|random}` replaces the
default growth order with a spatially defined one (`fp` = fovea-to-periphery,
`pf` = periphery-to-fovea within each batch). The mode name is appended to the
output filenames.

```bash
cd src
SHARED_DATA_ROOT=../data python experiment.py \
    --data R1_gpr_grid --tag lh --mode mds \
    --radius 1.30 --tangent 2.20 --custom_batch_mode polar_fp
```
