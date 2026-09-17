# Self-organization of higher visual areas across the cortical surface

Reference implementation of the growth model. Connectivity grows outward from V1
into higher visual areas (V2, V3, V4) on fMRI-derived cortical surface geometry,
and the retinotopic tuning of the higher areas is predicted from that
connectivity and compared against the measured maps.

The model is deterministic and every other random state is fixed, so the numbers
reported in the paper reproduce exactly on re-execution. The one exception is
`--custom_batch_mode *_random`, which draws a fresh permutation per call.

`--hierarchical` grows in sequential stages: a stage ends once 80% of its source
nodes have connected, and the targets it assigned become the sources of the next
stage (`src/hierarchical.py`).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10.12; `requirements.txt` pins the versions used for the
published results. A GPU is optional: the default run takes about 6 s on a GPU
and about 75 s on CPU.

## Run

```bash
bash scripts/run_example.sh
bash scripts/run_example.sh --data M1_gpr_grid --tag rh
```

With no arguments it runs the NMT template, left hemisphere, at the parameters
used for every result in the paper. The equivalent direct call, which also
exposes the remaining options:

```bash
cd src
SHARED_DATA_ROOT=../data python experiment.py --data NMT_gpr_grid --tag lh
```

| Option | Meaning | Default |
|---|---|---|
| `--data` | Subject: `NMT_gpr_grid`, `M1_gpr_grid` … `M6_gpr_grid` | `NMT_gpr_grid` |
| `--tag` | Hemisphere, `lh` or `rh` | `lh` |
| `--radius` | Radial kernel width sigma_R | `1.30`, or `1.40` with `--hierarchical` |
| `--tangent` | Tangential kernel width sigma_T | `2.20` |
| `--num_degree` | V1 parents per extrastriate node | `1` |
| `--custom_batch_mode` | Spatially defined growth order, `{angle\|polar\|euclidean\|x}_{fp\|pf\|random}` (`fp` = fovea to periphery) | off |
| `--hierarchical` | Grow in sequential stages instead of from V1 only | off |
| `--kernel` | Distance kernel: `polar`, or `sphere` for the smooth spherical surface | `polar` |
| `--no_plot` | Skip the comparison plot | off |
| `--video` | Also write the growth animation as HTML | off |

`SHARED_DATA_ROOT` sets the input directory. `COLOR_PHI_COVERAGE` (default
`0.85`) sets the fraction of the V1 phase range the display colour scale spans;
it affects figure colours only, never a reported number.

## Input

`data/{subject}_gpr_grid_{hemi}.pkl` is a dict keyed by node ID; each entry holds:

| Field | Description |
|---|---|
| `area` | Visual area label (1 = V1, 2 = V2, 3 = V3, 4 = V4) |
| `tuning` | 2D retinotopic tuning vector `[x, y]`, in visual degrees |
| `loc` | 2D MDS coordinates of the cortical node, used for the kernel |
| `is_center` | `1` for the foveal-center node, `0` otherwise |

`NMT` is the population template used for the primary analyses and `M1`–`M6` are
the six individual macaques, each with a left and a right hemisphere.
`NMTsphere_gpr_grid_{lh,rh}.pkl` is the template resampled on a smooth spherical
surface; run it with `--kernel sphere`, which measures distances along the sphere.
`data/NMT_{lh,rh}.pkl` are the native cortical **mesh** for the template, a
different node set carrying `loc_3D`, `loc_sphere` and `tuning_original`; do not
pass them to `--data`.

`scripts/prepare_input.py` builds a `_gpr_grid_` file for a new hemisphere from a
pairwise surface-distance matrix and the cortical data at those nodes (MDS
embedding, uniform-grid resampling, Gaussian-process interpolation of the
tuning). Computing the surface distances is an input to this pipeline, not a part
of it. See `--help` for the options.

## Output

Written under `outputs/` for the default run:

| File | Description |
|---|---|
| `outputs/predictions/mds/predicted_NMT_gpr_grid_lh_deterministic_1.30_2.20.tsv` | Predicted and measured tuning, one row per V2–V4 node |
| `outputs/predictions/mds/W_NMT_gpr_grid_lh_deterministic_1.30_2.20.npz` | Connection matrix `W`, plus `node_generation_order` and `batch_info` |
| `outputs/plots/NMT_gpr_grid_lh_tuning_compare_1.30_2.20.png` | Measured and predicted polar-angle and eccentricity maps |

The growth sequence at any intermediate step is reconstructed from `W` together
with `node_generation_order`; no per-step snapshot is stored.

Columns of the TSV:

| Column | Description |
|---|---|
| `Node_ID` | Key of the node in the input pkl |
| `Area` | Visual area label, 2 = V2, 3 = V3, 4 = V4 |
| `Pred_0`, `Pred_1` | Predicted tuning vector `[x, y]`, in visual degrees |
| `True_0`, `True_1` | Measured tuning vector `[x, y]`, in visual degrees |
| `Pred_polar_angle_deg`, `True_polar_angle_deg` | Polar angle in degrees |
| `Pred_eccentricity_deg`, `True_eccentricity_deg` | Eccentricity in visual degrees |
| `Retinotopic_error_deg` | Distance between the predicted and the measured tuning vector, in visual degrees |

Polar angle and eccentricity are the tuning vectors read in polar coordinates
about the V1 anchor, which `src/TUNING_COLOR_UTILS.py` derives from the V1 tuning
of that hemisphere (`_anchor_and_cy_from_v1`), not about the origin of `[x, y]`;
`tuning_to_polar` in the same file does the conversion. This is the frame every
polar angle and eccentricity in the paper is quoted in, and the angle increases
toward the upper visual field, so the two hemispheres are mirror images of each
other. The Spearman correlations quoted for Fig. 2 are rank correlations between
the `True_` and `Pred_` columns over all V2–V4 nodes of a hemisphere.

## Which command produces which figure

| Figure | Command |
|---|---|
| Fig. 2, maps and per-area correspondence | default run, `--tag lh` and `--tag rh` |
| Fig. 4, areal boundaries and map size | default run; the boundaries are read off polar angle progression lines traced by hand on the measured and on the predicted map |
| Fig. 5A, individual macaques | `--data M1_gpr_grid` … `M6_gpr_grid`, both hemispheres |
| Supp. Fig. S2, parameter grid | `--radius` and `--tangent` swept from 0.5 to 2.5 in steps of 0.1 |
| Supp. Fig. S4, batch ordering | `--custom_batch_mode euclidean_pf`, `euclidean_fp` and `euclidean_random` |
| Supp. Fig. S6, smooth-sphere control | `--data NMTsphere_gpr_grid --kernel sphere` |
| Supp. Fig. S7, hierarchical growth | `--hierarchical` |
| Supp. Fig. S9, individual macaques, right hemisphere | `--data M1_gpr_grid` … `M6_gpr_grid --tag rh` |
| Supp. Fig. S10, single-parameter kernel | `--radius` equal to `--tangent`, both 2.13 |

The other figures need an input or an analysis that is not distributed here: the
rotation control (Fig. 3) rotates the template's V1 tuning before the run, the
cross-monkey transfer (Fig. 5B) substitutes one animal's V1 into another's grid,
and the curvature (Supp. Fig. S5) and sampling-line (Supp. Fig. S8) controls need
cortical curvature and the traced lines respectively. The figure scripts themselves are likewise not part of
this repository; the values they plot are provided as Source Data with the paper.

## Citation

See `CITATION.cff`.
