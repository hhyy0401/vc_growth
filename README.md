# Geometric Constraints in the Development of Primate Extrastriate Visual Cortex

Reference implementation for the macaque retinotopy growth model.
The model simulates feed-forward connectivity growth from V1 into higher visual
areas (V2, V3, V4) on fMRI-derived cortical surface data, then predicts the
retinotopic tuning of the higher areas and compares it against ground truth.

## Repository layout

```
.
├── src/                       # model + I/O + plotting code (minimal set)
│   ├── experiment.py          # command-line entry point
│   ├── polarModel.py          # growth simulation (VisualMatrix3D)
│   ├── utils.py               # data loading, MSE, result/plot saving
│   ├── visualizationUtil.py   # optional animation helpers
│   ├── node_color_utils.py    # color helpers (re-exports colormap10)
│   ├── colormap10.py          # 10-bin polar-angle colormap
│   └── TUNING_COLOR_UTILS.py  # tuning-to-color mapping
├── data/                      # input pickles (bundled, ~3.5 MB)
│   ├── R1_gpr_grid_{lh,rh}.pkl     # NMT template macaque
│   └── S{1..6}_gpr_grid_{lh,rh}.pkl # six individual macaques
├── scripts/
│   └── run_example.sh         # one-command reproduction
├── outputs/                   # created at run time (git-ignored)
├── requirements.txt
└── README.md
```

## Input data

Each `data/{subject}_{hemi}.pkl` is a dict keyed by node ID; each entry holds:

| Field       | Description                                             |
|-------------|---------------------------------------------------------|
| `area`      | Visual area label (1 = V1, 2 = V2, 3 = V3, 4 = V4)      |
| `tuning`    | 2D retinotopic tuning vector `[x, y]` (normalized)      |
| `loc`       | 2D MDS coordinates `[x, y]` used for the kernel geometry|
| `is_center` | `1` for the foveal-center node, `0` otherwise           |

Subjects: `R1` (NMT template) and `S1`–`S6` (six individual macaques),
each with a left (`lh`) and right (`rh`) hemisphere = 14 files.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10. A GPU is optional; the model runs on CPU
(set the environment before running if you have several GPUs and want a
specific one, otherwise PyTorch picks a device automatically).

## Run the example

```bash
bash scripts/run_example.sh
bash scripts/run_example.sh --data S1_gpr_grid --tag rh
bash scripts/run_example.sh \
    --data R1_gpr_grid --sigma-r 1.30 --sigma-t 2.20
```

The script exposes the dataset, hemisphere, and two model parameters. With no
arguments it runs R1 LH with **sigma_R = 1.30** and **sigma_T = 2.20**.
The model uses the published default **alpha = 0.30**.

## Running directly

```bash
cd src
SHARED_DATA_ROOT=../data python experiment.py \
    --data R1_gpr_grid --tag lh \
    --mode mds --distance_mode polar --algo deterministic \
    --radius 1.30 --tangent 2.20
```

### `run_example.sh` arguments

| Argument      | Meaning                                    | Default       |
|---------------|--------------------------------------------|---------------|
| `--data`      | Dataset (`R1_gpr_grid`, `S1_gpr_grid`, …) | `R1_gpr_grid` |
| `--tag`       | Hemisphere (`lh` / `rh`)                   | `lh`          |
| `--sigma-r`   | Radial kernel width (sigma_R)              | `1.30`        |
| `--sigma-t`   | Tangential kernel width (sigma_T)          | `2.20`        |

The wrapper keeps `alpha=0.30`, `mode=mds`, `distance_mode=polar`, and
`algo=deterministic` at their published defaults.

## Output

For the default run, the script creates:

| File | Description |
|------|-------------|
| `outputs/predictions/mds/predicted_R1_gpr_grid_lh_deterministic_1.30_2.20_0.30.tsv` | Predicted and empirical V2–V4 tuning values |
| `outputs/predictions/mds/W_R1_gpr_grid_lh_deterministic_1.30_2.20_0.30.npz` | Model weight matrix and node-generation order |
| `outputs/plots/R1_gpr_grid_lh_tuning_compare_1.30_2.20_0.30.png` | Empirical and predicted polar-angle and eccentricity maps |
