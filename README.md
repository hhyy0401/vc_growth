# Self-organization of higher visual areas across the cortical surface

Reference implementation of the growth model. Connectivity grows outward from V1
into higher visual areas (V2, V3, V4) on fMRI-derived cortical surface geometry,
and the retinotopic tuning of the higher areas is predicted from that connectivity
and compared against the measured maps.

The model is deterministic and every other random state in the pipeline is fixed,
so the numbers reported in the paper reproduce exactly on re-execution. The one
exception is noted under [Batch-ordering control](#batch-ordering-control).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with **Python 3.10.12**. `requirements.txt` pins every dependency to the
version used to produce the published results:

| Package | Version | | Package | Version |
|---|---|---|---|---|
| numpy | 1.26.4 | | matplotlib | 3.8.3 |
| pandas | 2.2.1 | | scikit-learn | 1.4.1.post1 |
| scipy | 1.12.0 | | plotly | 5.24.1 |
| torch | 2.2.1 | | tqdm | 4.66.2 |
| scikit-optimize | 0.10.1 | | | |

PyTorch is CPU-capable; the CUDA build (2.2.1+cu121) was used but is not required.
`scikit-optimize` is needed only for `--param_search search`. A GPU is optional:
the default run takes about 6 s on a GPU and about 75 s on CPU.

## Run

```bash
bash scripts/run_example.sh
bash scripts/run_example.sh --data M1_gpr_grid --tag rh
bash scripts/run_example.sh --data NMT_gpr_grid --tag lh --sigma-r 1.30 --sigma-t 2.20
```

With no arguments it runs the NMT template, left hemisphere, at sigma_R = 1.30 and
sigma_T = 2.20, which is the parameter pair used for every result in the paper.

`run_example.sh` exposes the dataset, the hemisphere and the model's two free
parameters, and keeps `mode=mds`, `distance_mode=polar` and `algo=deterministic`
fixed:

| Option | Meaning | Default |
|--------------|-----------------------------------|---------------|
| `--data`     | Subject                           | `NMT_gpr_grid` |
| `--tag`      | Hemisphere (`lh` / `rh`)          | `lh`          |
| `--sigma-r`  | Radial kernel width (sigma_R)     | `1.30`        |
| `--sigma-t`  | Tangential kernel width (sigma_T) | `2.20`        |

Equivalent direct call, where the same four appear as `--data`, `--tag`,
`--radius` and `--tangent`:

```bash
cd src
SHARED_DATA_ROOT=../data python experiment.py \
    --data NMT_gpr_grid --tag lh \
    --mode mds --distance_mode polar --algo deterministic \
    --radius 1.30 --tangent 2.20
```

Run from inside `src/`. Output paths are relative to the working directory, so
calling `python src/experiment.py` from the repository root writes outside the
repository. `experiment.py --help` lists further flags used during development;
the four above are the ones needed to reproduce the paper.

Two environment variables change behaviour and are read directly:
`SHARED_DATA_ROOT` sets the input directory, and `COLOR_PHI_COVERAGE` (default
`0.85`) sets the fraction of the V1 phase range the display colour scale spans. The
second affects figure colours only, never a reported number.

## Input

`data/{subject}_gpr_grid_{hemi}.pkl` is a dict keyed by node ID; each entry holds:

| Field       | Description                                              |
|-------------|----------------------------------------------------------|
| `area`      | Visual area label (1 = V1, 2 = V2, 3 = V3, 4 = V4)       |
| `tuning`    | 2D retinotopic tuning vector `[x, y]`, in visual degrees |
| `loc`       | 2D MDS coordinates `[x, y]` used for the kernel geometry |
| `is_center` | `1` for the foveal-center node, `0` otherwise            |

Subjects: `NMT` is the population template used for the primary analyses. `M1`–`M6`
are the six individual macaques. The labels match the ones printed in the paper.
Each has a left (`lh`) and a right (`rh`) hemisphere, so 14 files, and all 14 are
included here.

`data/NMT_lh.pkl` and `data/NMT_rh.pkl` are also present. These are the native
cortical **mesh** for the template, not the resampled grid the model runs on: a
different node set (3,769 and 3,814 nodes against 3,486 and 3,238) carrying the
extra fields `loc_3D`, `loc_sphere` and `tuning_original`. They are the substrate
for the phase-versus-distance analysis in Fig. 4, where geodesic distance has to be
measured on the mesh. Do not pass them to `--data`; use the `_gpr_grid_` files.

Nodes are re-sorted by area when loaded, so `Node_ID` in the output is the original
pkl key rather than a row index.

## Preparing a new dataset

`scripts/prepare_input.py` builds a `_gpr_grid_` file from a new hemisphere, so the
model can be applied to data other than the macaques shipped here.

**Computing the cortical-surface distance matrix is an input to this pipeline, not
a part of it.** It depends on which surface representation the dataset uses and on
the software that walks that surface (SUMA/SurfDist, FreeSurfer, pycortex, ...), so
it has to be produced beforehand with the tools that fit the data at hand.

### Required inputs

| Input | Contents |
|---|---|
| `--distances` | Pairwise surface distance over the N cortical nodes of one hemisphere: an `(N, N)` `.npy`/`.npz` array, or its condensed upper-triangle form |
| `--cortex` | The cortical data at those same nodes: a `.csv`/`.txt` with a header, or an `.npz`, holding `node`, `area`, `polar_angle` and `eccentricity` |
| `--out` | Path of the `.pkl` to write |

`area` uses the same labels as the shipped data (1 = V1 … 4 = V4). `polar_angle` is
in degrees unless `--radians` is given, and together with `eccentricity` it forms
the `tuning` vector the model compares against. Rows of `--cortex` must line up with
rows of `--distances`; if the matrix covers a wider or differently ordered node set,
pass `--distance-nodes` with the node index of each matrix row and it will be subset
and reordered to match.

### Usage

```bash
python scripts/prepare_input.py \
    --distances distances_lh.npy \
    --cortex    cortex_lh.csv \
    --out       data/X1_gpr_grid_lh.pkl \
    --tag       lh
```

Then run the model on it exactly as on the shipped subjects:

```bash
bash scripts/run_example.sh --data X1_gpr_grid --tag lh
```

Before the first stage, non-V1 tuning that falls outside the range of the V1
tuning is clamped to that range. The model predicts a higher-area node's tuning as
a weighted average of the V1 tunings it connects to, so anything outside the V1
range is unreachable by construction and would only ever register as error; the
distributed datasets are clipped the same way. `--no-clip-tuning` leaves it alone.

The three stages, and the options that control them:

1. **2D MDS embedding** of the distance matrix, flattening the folded patch into a
   plane. `--mds-n-init`, `--mds-max-iter`.
2. **Uniform-grid resampling.** Cortical nodes are unevenly spaced, which biases the
   growth order, so the MDS plane is resampled onto an axis-aligned lattice.
   `--spacing` (default 0.75, in the units of the distance matrix), `--epsilon`
   (how close a cortical node must be for a lattice point to be kept), `--pad-frac`,
   `--contamination`, `--dbscan-eps`, `--dbscan-min-samples`. `--epsilon` and
   `--dbscan-eps` default to values scaled to the node density of the input, so the
   same settings work on sparser surfaces than the macaque ones.
3. **Matern Gaussian-process interpolation** of the tuning onto the lattice, fitted
   one visual-field axis at a time. `--matern-nu` (default 2.5),
   `--matern-length-scale` (default 1.0), `--noise-level` (default 0.1). Area labels
   are categorical and are carried over by nearest neighbour instead.

`--seed` (default 42) fixes the MDS, outlier-detection and Gaussian-process random
states; the script is deterministic for a given seed and set of inputs.
`scripts/prepare_input.py --help` lists everything.

### Generated output

A single `.pkl` at `--out`, in exactly the format described under
[Input](#input): a dict keyed by contiguous integers `0 … N-1`, each entry holding
`loc`, `tuning`, `area` and `is_center`. The script also locates the foveal
confluence (the V1 node on the mid-V1 phase line closest to the V1 border) and flags
it as `is_center`, which is what orients the model's radial/tangential kernel; it
stops with an error if that cannot be found.

Note that MDS fixes a configuration only up to rotation and reflection, so a
rebuilt file is not expected to be byte-identical to a shipped one even from the
same surface. Rebuilding the NMT left hemisphere from its geodesic distance matrix
gives 3,435 grid nodes against the shipped 3,486, with matching extent, area
composition and eccentricity range.

## Output

Written under `outputs/` (git-ignored, and not included in the archived release
because everything in it is regenerated by the commands above). For the default run:

| File | Description |
|------|-------------|
| `outputs/predictions/mds/predicted_NMT_gpr_grid_lh_deterministic_1.30_2.20.tsv` | Predicted and empirical V2–V4 tuning values, one row per node |
| `outputs/predictions/mds/W_NMT_gpr_grid_lh_deterministic_1.30_2.20.npz` | Connection matrix `W`, plus `node_generation_order` and `batch_info` |
| `outputs/plots/NMT_gpr_grid_lh_tuning_compare_1.30_2.20.png` | Empirical and predicted polar-angle and eccentricity maps |

The growth sequence at any intermediate step is reconstructed from `W` together
with `node_generation_order`; no per-step snapshot is stored.

## Which command produces which figure

| Figure | Command |
|---|---|
| Fig. 2, maps and per-area correspondence | default run, `--data NMT_gpr_grid --tag lh` and `--tag rh` |
| Fig. 4, phase versus geodesic distance | default run; the analysis reads `NMT_{lh,rh}.pkl` for mesh distances |
| Fig. 5A, individual macaques | `--data M1_gpr_grid` … `M6_gpr_grid`, both hemispheres |
| Supp. Fig. S2, parameter grid | `--radius` and `--tangent` swept from 0.5 to 2.5 in steps of 0.1 |
| Supp. Fig. S4, batch ordering | `--custom_batch_mode`, see below |
| Supp. Fig. S9, isotropic kernel | `--radius` equal to `--tangent` |

Three further analyses run this same model on inputs derived from the files above
rather than on the files themselves, and those derived inputs are not distributed
here: the rotation control (Fig. 3) rotates the template's MDS coordinates, the
cross-monkey transfer (Fig. 5B) substitutes one animal's V1 into another's grid, and
the hierarchical variant (Supp. Fig. S7) uses a separate entry point. The figure
scripts themselves are likewise not part of this repository; the values they plot
are provided as Source Data with the paper.

## Batch-ordering control

`--custom_batch_mode {angle|polar|euclidean|x}_{fp|pf|random}` replaces the default
growth order with a spatially defined one (`fp` = fovea to periphery, `pf` =
periphery to fovea within each batch). The mode name is appended to the output
filenames.

```bash
cd src
SHARED_DATA_ROOT=../data python experiment.py \
    --data NMT_gpr_grid --tag lh --mode mds \
    --radius 1.30 --tangent 2.20 --custom_batch_mode polar_fp
```

The `_random` orders draw a fresh permutation on every call and are not seeded, so
they reproduce the reported behaviour but not the exact stored output. The
deterministic orders (`_fp`, `_pf`) and every other result in the paper reproduce
byte for byte.

## Citation

See `CITATION.cff`.
