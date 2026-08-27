#!/usr/bin/env python3
"""Convert a cortical-surface distance matrix plus retinotopy into a model input.

The growth model runs on a flat, uniformly sampled sheet of nodes: `data/*_gpr_grid_*.pkl`.
This script builds one of those files from the two things a retinotopy dataset
normally provides:

  1. a precomputed pairwise distance matrix over the cortical nodes of one
     hemisphere (geodesic distance along the surface, in mm), and
  2. the cortical data measured at those same nodes -- polar angle, eccentricity
     and a visual-area label.

Computing the distance matrix itself is *not* part of this pipeline. It depends on
the cortical-surface representation and on the software used to walk that surface
(SUMA/SurfDist, FreeSurfer, pycortex, ...), so it is treated as an input here.

Stages, in order:

  1. **2D MDS embedding.** Metric multidimensional scaling of the distance matrix
     flattens the folded patch into a plane while preserving surface distance as
     well as two dimensions allow.
  2. **Uniform-grid resampling.** Cortical nodes are unevenly spaced, which biases
     the growth order. The MDS plane is resampled onto an axis-aligned lattice of
     fixed spacing, keeping only lattice points that fall inside the cortical
     patch (plus enclosed holes, which are filled).
  3. **Matern Gaussian-process interpolation.** Tuning is carried from the
     scattered cortical nodes onto the lattice, one visual-field axis at a time.
     Area labels are categorical and are carried over by nearest neighbour instead.

Finally the foveal confluence is located (the V1 node on the mid-V1 phase line
closest to the V1 border) and flagged as `is_center`, which is what the model uses
to orient its radial/tangential kernel.

Usage
-----
    python scripts/prepare_input.py \
        --distances  distances_lh.npy \
        --cortex     cortex_lh.csv \
        --out        data/X1_gpr_grid_lh.pkl \
        --tag        lh
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.manifold import MDS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from TUNING_COLOR_UTILS import compute_tuning_colors  # noqa: E402

CORTEX_COLUMNS = ("node", "area", "polar_angle", "eccentricity")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def load_distance_matrix(path, key=None):
    """Load an (N, N) distance matrix from .npy or .npz.

    A condensed upper-triangle vector (as produced by `scipy.spatial.distance.pdist`
    or stored by SurfDist-style tools) is expanded to the square form.
    """
    if path.endswith(".npz"):
        with np.load(path) as handle:
            names = list(handle.keys())
            if key is None:
                for candidate in ("distances", "distance_matrix", "cond", "D"):
                    if candidate in names:
                        key = candidate
                        break
            if key is None:
                if len(names) != 1:
                    raise ValueError(
                        f"{path} holds {names}; pick one with --distances-key"
                    )
                key = names[0]
            array = np.asarray(handle[key])
    else:
        array = np.asarray(np.load(path))

    array = array.astype(np.float64, copy=False)
    if array.ndim == 1:
        array = squareform(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{path}: expected a square or condensed matrix, got {array.shape}")

    # MDS needs an exactly symmetric, zero-diagonal dissimilarity.
    array = 0.5 * (array + array.T)
    np.fill_diagonal(array, 0.0)
    return array


def load_row_nodes(path, key=None):
    """Load the node index of each distance-matrix row."""
    if path.endswith(".npz"):
        with np.load(path) as handle:
            names = list(handle.keys())
            if key is None:
                for candidate in ("node", "nodes", "node_idx", "node_ids"):
                    if candidate in names:
                        key = candidate
                        break
            if key is None:
                raise ValueError(f"{path} holds {names}; pick one with --distance-nodes-key")
            return np.asarray(handle[key]).ravel().astype(np.int64)
    return np.asarray(np.load(path)).ravel().astype(np.int64)


def align_distances(distances, row_nodes, node_ids):
    """Subset and reorder the distance matrix to match the cortex table's nodes."""
    position = {int(n): i for i, n in enumerate(row_nodes)}
    missing = [int(n) for n in node_ids if int(n) not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} node(s) in the cortex table have no row in the distance "
            f"matrix, e.g. {missing[:5]}"
        )
    order = np.array([position[int(n)] for n in node_ids], dtype=np.int64)
    return distances[np.ix_(order, order)]


def load_cortex_table(path):
    """Load the per-node cortical data as (node_ids, areas, polar_angle, eccentricity).

    Accepts a delimited text file with a header naming the columns in
    `CORTEX_COLUMNS`, or an .npz holding one array per column.
    """
    if path.endswith(".npz"):
        with np.load(path) as handle:
            missing = [c for c in CORTEX_COLUMNS if c not in handle]
            if missing:
                raise ValueError(f"{path} is missing array(s) {missing}")
            columns = {c: np.asarray(handle[c]).ravel() for c in CORTEX_COLUMNS}
    else:
        table = np.genfromtxt(path, delimiter=None if path.endswith(".txt") else ",",
                              names=True, dtype=None, encoding="utf-8")
        missing = [c for c in CORTEX_COLUMNS if c not in (table.dtype.names or ())]
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {missing}; expected a header with "
                f"{', '.join(CORTEX_COLUMNS)}"
            )
        columns = {c: np.asarray(table[c]).ravel() for c in CORTEX_COLUMNS}

    node_ids = columns["node"].astype(np.int64)
    areas = columns["area"].astype(np.int64)
    polar_angle = columns["polar_angle"].astype(np.float64)
    eccentricity = columns["eccentricity"].astype(np.float64)
    return node_ids, areas, polar_angle, eccentricity


def tuning_from_retinotopy(polar_angle, eccentricity, degrees=True):
    """Polar angle + eccentricity -> the (x, y) visual-field vector the model stores."""
    theta = np.deg2rad(polar_angle) if degrees else np.asarray(polar_angle, dtype=float)
    return np.column_stack([eccentricity * np.cos(theta), eccentricity * np.sin(theta)])


# ---------------------------------------------------------------------------
# Stage 1: MDS
# ---------------------------------------------------------------------------
def mds_embed(distances, seed=42, n_init=4, max_iter=300):
    """Metric MDS of a precomputed distance matrix onto the plane.

    The embedding is stored as produced: SMACOF centres it, but no rescaling,
    rotation or reflection is applied here. The model applies its own alignment
    (V1 principal axis onto x, foveal confluence at the origin) when it loads the
    file, so the stored frame does not have to be canonical.
    """
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        normalized_stress=False,
    )
    coords = mds.fit_transform(distances)
    return coords, float(mds.stress_)


# ---------------------------------------------------------------------------
# Stage 2: uniform grid
# ---------------------------------------------------------------------------
def drop_outliers(xy, tuning, contamination=0.05, seed=42):
    """IsolationForest over (x, y, tuning_x, tuning_y); returns an inlier mask."""
    if contamination <= 0:
        return np.ones(len(xy), dtype=bool)
    forest = IsolationForest(contamination=contamination, random_state=seed)
    return forest.fit_predict(np.hstack([xy, tuning])) == 1


def keep_largest_island(xy, eps, min_samples=10):
    """Keep the largest DBSCAN cluster, dropping detached fragments of the embedding."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    named = labels[labels != -1]
    if named.size == 0:
        return np.ones(len(xy), dtype=bool)
    values, counts = np.unique(named, return_counts=True)
    return labels == values[int(np.argmax(counts))]


def uniform_grid(xy, spacing, epsilon, pad_frac=0.10):
    """Lattice points of the given spacing that lie inside the embedded patch.

    A lattice point is kept when a cortical node sits within `epsilon` of it.
    Enclosed gaps left by that test are filled back in, so the result is a solid
    region rather than a perforated one.
    """
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_pad = (x_max - x_min) * pad_frac
    y_pad = (y_max - y_min) * pad_frac

    xs = np.arange(x_min - x_pad, x_max + x_pad + spacing, spacing)
    ys = np.arange(y_min - y_pad, y_max + y_pad + spacing, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    nearest, _ = cKDTree(xy).query(candidates, k=1)
    inside = binary_fill_holes((nearest <= epsilon).reshape(len(ys), len(xs)))
    return candidates[inside.ravel()]


def density_scaled(xy, ratio, floor):
    """Scale a length tuned on one node density to the density actually present.

    The reference values (`epsilon` = 1.0, DBSCAN `eps` = 3.0) were set on macaque
    surfaces whose median nearest-neighbour spacing is ~0.59 mm. On sparser data
    those absolute lengths would discard most of the cortex, so they are expressed
    as a multiple of the observed spacing and floored at the reference value.
    """
    nearest = cKDTree(xy).query(xy, k=2)[0][:, 1]
    return max(floor, ratio * float(np.median(nearest)))


# ---------------------------------------------------------------------------
# Stage 3: Gaussian-process interpolation
# ---------------------------------------------------------------------------
def gp_interpolate(xy, values, query, nu=2.5, length_scale=1.0, noise_level=0.1, seed=42):
    """Interpolate each visual-field axis onto `query` with a Matern GP.

    The two axes are fitted independently. `normalize_y` centres and scales the
    target, so the unit-amplitude kernel fits data of any tuning amplitude instead
    of absorbing a large signal into the white-noise term.
    """
    kernel = Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise_level)
    out = np.empty((len(query), values.shape[1]), dtype=float)
    for axis in range(values.shape[1]):
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True)
        gpr.fit(xy, values[:, axis])
        out[:, axis] = gpr.predict(query)
    return out


def nearest_area(query, xy, areas):
    """Area labels are categorical, so they are copied from the nearest cortical node."""
    _, index = cKDTree(xy).query(query, k=1)
    return np.asarray(areas)[index]


# ---------------------------------------------------------------------------
# Foveal confluence
# ---------------------------------------------------------------------------
def estimate_center(xy, tuning, areas, tag=None):
    """Locate the foveal confluence on the resampled grid.

    The V1 nodes whose phase sits at the low end of the V1 range trace the line
    that runs from the fovea out along the middle of V1. Fitting that line and
    walking it to where it meets the V1 border gives the foveal confluence, which
    the model uses as the origin of its radial direction.
    """
    areas = np.asarray(areas, dtype=int)
    v1 = areas == 1
    if not np.any(v1) or not np.any(areas != 1):
        return None

    v1_xy = xy[v1]
    colors = compute_tuning_colors(tuning[v1], v1_mask=np.ones(int(v1.sum()), dtype=bool), tag=tag)

    line_nodes = v1_xy[colors < 0.05]
    if len(line_nodes) < 2:
        return None

    # Principal axis of the mid-V1 phase line, and the rotation that lays it flat.
    line_mean = line_nodes.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((line_nodes - line_mean).T))
    direction = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.arctan2(direction[1], direction[0])
    rotation = np.array([[np.cos(-angle), -np.sin(-angle)],
                         [np.sin(-angle), np.cos(-angle)]])

    # V1 nodes with at least one non-V1 neighbour form the V1 border.
    _, neighbours = cKDTree(xy).query(v1_xy, k=5)
    neighbours = neighbours[:, 1:]
    on_border = [i for i, idx in enumerate(neighbours)
                 if np.any(areas[idx] == 1) and np.any(areas[idx] != 1)]
    if len(on_border) < 4:
        return None

    border_rotated = (v1_xy[on_border] - line_mean) @ rotation.T
    closest = int(np.argmin(np.abs(border_rotated[:, 1])))
    center = np.array([border_rotated[closest, 0], 0.0]) @ rotation + line_mean
    return float(center[0]), float(center[1])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build(distances, areas, tuning, args):
    if not (len(areas) == len(tuning) == len(distances)):
        raise ValueError(
            f"node count mismatch: distances {len(distances)}, areas {len(areas)}, "
            f"tuning {len(tuning)}"
        )

    print(f"[1/3] MDS embedding of {len(distances)} nodes ...")
    xy, stress = mds_embed(distances, seed=args.seed, n_init=args.mds_n_init,
                           max_iter=args.mds_max_iter)
    print(f"      stress = {stress:.4f}")

    inliers = drop_outliers(xy, tuning, contamination=args.contamination, seed=args.seed)
    xy, tuning, areas = xy[inliers], tuning[inliers], areas[inliers]
    dbscan_eps = args.dbscan_eps if args.dbscan_eps is not None else density_scaled(xy, 5.1, 3.0)
    island = keep_largest_island(xy, eps=dbscan_eps, min_samples=args.dbscan_min_samples)
    xy, tuning, areas = xy[island], tuning[island], areas[island]
    print(f"      kept {len(xy)} nodes (DBSCAN eps={dbscan_eps:.2f})")

    epsilon = args.epsilon if args.epsilon is not None else density_scaled(xy, 1.70, 1.0)
    print(f"[2/3] resampling onto a {args.spacing} lattice (epsilon={epsilon:.2f}) ...")
    grid = uniform_grid(xy, spacing=args.spacing, epsilon=epsilon, pad_frac=args.pad_frac)
    print(f"      {len(grid)} grid nodes")

    print(f"[3/3] Matern GP interpolation (nu={args.matern_nu}, "
          f"length_scale={args.matern_length_scale}) ...")
    grid_tuning = gp_interpolate(xy, tuning, grid, nu=args.matern_nu,
                                 length_scale=args.matern_length_scale,
                                 noise_level=args.noise_level, seed=args.seed)
    grid_areas = nearest_area(grid, xy, areas)

    center = estimate_center(grid, grid_tuning, grid_areas, tag=args.tag)
    if center is None:
        raise RuntimeError(
            "could not locate the foveal confluence; check that the area labels "
            "use 1 = V1 and that V1 is present in the input"
        )
    center_index = int(cKDTree(grid).query(np.asarray(center)[None, :], k=1)[1][0])
    print(f"      foveal confluence at ({center[0]:.4f}, {center[1]:.4f}) -> node {center_index}")

    return {
        i: {
            "loc": (float(grid[i, 0]), float(grid[i, 1]), 0.0),
            "tuning": (float(grid_tuning[i, 0]), float(grid_tuning[i, 1])),
            "area": int(grid_areas[i]),
            "is_center": 1 if i == center_index else 0,
        }
        for i in range(len(grid))
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distances", required=True,
                        help="(N, N) surface-distance matrix, or its condensed form (.npy/.npz)")
    parser.add_argument("--distances-key", default=None,
                        help="Array name to read when --distances is an .npz with several arrays")
    parser.add_argument("--distance-nodes", default=None,
                        help="Node index of each distance-matrix row (.npy/.npz). Give this when "
                             "the matrix covers more nodes, or a different order, than --cortex; "
                             "the matrix is then subset and reordered to match")
    parser.add_argument("--distance-nodes-key", default=None,
                        help="Array name to read when --distance-nodes is an .npz with several arrays")
    parser.add_argument("--cortex", required=True,
                        help="Per-node cortical data (.csv/.txt with a header, or .npz) holding "
                             + ", ".join(CORTEX_COLUMNS))
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--tag", default="lh", choices=["lh", "rh"],
                        help="Hemisphere; sets which side the phase colour scale is anchored on")
    parser.add_argument("--radians", action="store_true",
                        help="Polar angle is in radians (default: degrees)")

    grid = parser.add_argument_group("grid resampling")
    grid.add_argument("--spacing", type=float, default=0.75,
                      help="Lattice spacing in the units of the distance matrix (default: 0.75 mm)")
    grid.add_argument("--epsilon", type=float, default=None,
                      help="Keep a lattice point only if a node lies within this distance "
                           "(default: scaled to the node density, floored at 1.0)")
    grid.add_argument("--pad-frac", type=float, default=0.10,
                      help="Fraction of the embedding extent added as a margin (default: 0.10)")
    grid.add_argument("--contamination", type=float, default=0.05,
                      help="IsolationForest outlier fraction; 0 disables (default: 0.05)")
    grid.add_argument("--dbscan-eps", type=float, default=None,
                      help="DBSCAN radius for island removal "
                           "(default: scaled to the node density, floored at 3.0)")
    grid.add_argument("--dbscan-min-samples", type=int, default=10,
                      help="DBSCAN min_samples for island removal (default: 10)")

    model = parser.add_argument_group("MDS and Gaussian process")
    model.add_argument("--mds-n-init", type=int, default=4,
                       help="MDS restarts; the lowest-stress one is kept (default: 4)")
    model.add_argument("--mds-max-iter", type=int, default=300,
                       help="SMACOF iterations per MDS restart (default: 300)")
    model.add_argument("--matern-nu", type=float, default=2.5,
                       help="Matern smoothness nu (default: 2.5)")
    model.add_argument("--matern-length-scale", type=float, default=1.0,
                       help="Matern length scale (default: 1.0)")
    model.add_argument("--noise-level", type=float, default=0.1,
                       help="WhiteKernel noise level added to the Matern kernel (default: 0.1)")
    model.add_argument("--seed", type=int, default=42,
                       help="Random seed for MDS, IsolationForest and the GP (default: 42)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    distances = load_distance_matrix(args.distances, key=args.distances_key)
    node_ids, areas, polar_angle, eccentricity = load_cortex_table(args.cortex)
    tuning = tuning_from_retinotopy(polar_angle, eccentricity, degrees=not args.radians)

    if args.distance_nodes is not None:
        row_nodes = load_row_nodes(args.distance_nodes, key=args.distance_nodes_key)
        if len(row_nodes) != len(distances):
            raise ValueError(
                f"--distance-nodes has {len(row_nodes)} entries but the matrix has "
                f"{len(distances)} rows"
            )
        distances = align_distances(distances, row_nodes, node_ids)

    data = build(distances, areas, tuning, args)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(data, handle)
    print(f"Wrote {args.out}: {len(data)} nodes")


if __name__ == "__main__":
    main()
