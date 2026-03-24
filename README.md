# Geometric Constraints in the Development of Primate Extrastriate Visual Cortex

Code for the paper: [Geometric Constraints in the Development of Primate Extrastriate Visual Cortex](https://www.biorxiv.org/content/10.64898/2026.02.04.703881v1)

Simulates connectivity growth from V1 to higher visual areas (V2, V3, V4) using kernel-based propagation models on fMRI-derived cortical surface data.

Two models are provided:
- **`code/`** &mdash; Single-stage model with optional custom spatial batching.
- **`code_hierarchical/`** &mdash; Hierarchical (multi-stage) model where assigned targets become sources for the next stage.

## Input

Each run expects a pickle file at `data/{data}_{tag}.pkl` (relative to the repo root).

The pickle contains a dictionary keyed by node ID, where each entry has:

| Field | Description |
|---|---|
| `area` | Visual area label (1 = V1, 2 = V2, 3 = V3, 4 = V4) |
| `tuning` | 2D tuning preference vector `[x, y]` (normalized) |
| `loc` | 2D MDS coordinates `[x, y]` |
| `is_center` | `1` for the foveal center node, `0` otherwise |

> **Note:** Data files are not included in this repository.

## Output

Results are written to `results/` or `results_hierarchical/`:

| File | Description |
|---|---|
| `predicted_{data}_{tag}_{algo}_{params}.tsv` | Predicted vs. true tuning vectors for V2-V4 nodes |
| `W_{data}_{tag}_{algo}_{params}.npz` | Weight matrix `W` (V1 x N), node generation order, batch info |
| `{data}_{tag}_tuning_compare_{params}.png` | Three-panel comparison plot (true / predicted polar angle / eccentricity) |

Optional video animations (HTML) are saved under `plot_mirror/`.

## How It Works

### Model (`code/`)

1. **Build kernel matrix**: computes a pairwise rotated elliptical connectivity kernel between all nodes, decomposing displacement into local radial/tangential components with separate kernel parameters (`radius`, `tangent`).
2. **Iterative assignment**: in each step, an unassigned V2-V4 node is selected based on the highest propagation score (indirect signal through already-connected nodes + direct kernel affinity, weighted by source resource). The selected node is connected to its top-scoring V1 parent(s).
3. **Evaluate**: predicted V2-V4 tuning = (normalized W)^T @ V1 tuning. MSE is computed against ground truth.

### Hierarchical Model (`code_hierarchical/`)

Same kernel and scoring logic, but assignment proceeds in **stages**:
1. Stage 0 sources = V1 nodes, targets = all V2-V4 nodes.
2. Within a stage, nodes are assigned until `stage_ratio` of sources have out-degree >= 1.
3. Assigned targets with in-degree <= `in_degree_max` become sources for the next stage; unassigned targets carry over.
4. Each stage re-initializes its own W_stage; edges are projected back to root V1 nodes in the global weight matrix.

## Usage

### Model

```bash
bash run.sh
```

or directly:

```bash
cd code/
python experiment.py \
    --data R1_gpr_grid --tag lh \
    --radius 2.0 --tangent 2.0
```

### Hierarchical model

```bash
bash run_hier.sh
```

or directly:

```bash
cd code_hierarchical/
python experiment.py \
    --data R1_gpr_grid --tag lh \
    --hierarchical \
    --radius 2.0 --tangent 2.0
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset identifier (e.g., `R1_gpr_grid`, `S1_gpr_grid`) | `R1_gpr_grid` |
| `--tag` | Hemisphere (`lh` or `rh`) | `lh` |
| `--radius` | Radial kernel parameter | `2.0` |
| `--tangent` | Tangential kernel parameter (degrees) | `2.0` |
| `--action` | `run` (simulation) or `video` (run + animation) | `run` |
| `--stage-ratio` | Fraction of sources that must connect before advancing stage (hier only) | `0.90` |

## Dependencies

- Python 3.8+
- PyTorch
- NumPy, Pandas, SciPy
- Matplotlib, Plotly
- scikit-learn (for custom batch boundary detection)
- scikit-optimize (optional, for parameter search)
- tqdm
