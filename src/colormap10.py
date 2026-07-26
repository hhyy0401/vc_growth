"""Unified 10-bin color module (single source of truth for the whole repo).

All tuning / polar / eccentricity coloring goes through this module, and
everything is 10-bin. Consumers should not redefine colormaps or binning
locally; import from here. ``node_color_utils.py`` is a thin shim that
re-exports this module (``from colormap10 import *``).
"""

import os
import csv
import pickle

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

try:
    import matplotlib.pyplot as plt
except Exception:  # headless without pyplot still gives the discrete maps below
    plt = None

# ---------------------------------------------------------------------------
# Canonical bin count. Everything in this module is 10-bin, by design.
# ---------------------------------------------------------------------------
N_BINS = 10

# 10-bin polar-angle colormap (rainbow).
TUN10 = [
    "#73141B", "#fd4405", "#fe9800", "#fdff00", "#08fe01",
    "#33cd32", "#00fefe", "#0096ff", "#0143ff", "#4B0082",
]
CMAP_POLAR = ListedColormap(TUN10)

# 10-bin eccentricity colormap (sequential, distinct from polar).
if plt is not None:
    CMAP_ECC = ListedColormap(plt.cm.viridis(np.linspace(0, 1, N_BINS)))
else:  # pragma: no cover
    CMAP_ECC = ListedColormap([str(v) for v in np.linspace(0.1, 0.9, N_BINS)])

# BoundaryNorm mapping continuous [0, 1] values onto the 10 discrete bins.
NORM10 = BoundaryNorm(np.linspace(0, 1, N_BINS + 1), N_BINS)

# Back-compat aliases (TUN10/CMAP_P/CMAP_E/NORM).
CMAP_P = CMAP_POLAR
CMAP_E = CMAP_ECC
NORM = NORM10


def getColorMap():
    """Green->aqua->pink->red tuning colormap."""
    colors = ["green", "aqua", "pink", "red"]
    positions = np.linspace(0, 1, len(colors))
    return LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)))


# ---------------------------------------------------------------------------
# Quantile binning.
# ---------------------------------------------------------------------------
def quantile_bins(values, ref_values=None, mask=None, n_bins=N_BINS):
    """Assign each value to one of ``n_bins`` quantile bins.

    ``ref_values`` (default: ``values``) define the quantile edges; ``mask``
    optionally restricts which reference entries set the edges (e.g. V1 nodes).
    Returns an int array of bin indices in ``[0, n_bins - 1]``.
    """
    values = np.asarray(values, dtype=float)
    ref = values if ref_values is None else np.asarray(ref_values, dtype=float)
    if mask is not None:
        ref = ref[np.asarray(mask, dtype=bool)]
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros(values.shape, dtype=int)
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(values, edges[1:-1], right=False)
    return np.clip(idx, 0, n_bins - 1).astype(int)


# V1-referenced alias (quantile edges taken from V1 nodes only).
def quantile_bins_v1(values, v1_mask, n_bins=N_BINS):
    return quantile_bins(values, ref_values=values, mask=v1_mask, n_bins=n_bins)


# ---------------------------------------------------------------------------
# Tuning-vector coloring (10-bin).
# ---------------------------------------------------------------------------
def restore_tuning_range(tuning_vec):
    """Restore tuning values from [0,1] range to original range."""
    if tuning_vec is None:
        return None
    if not isinstance(tuning_vec, np.ndarray) or len(tuning_vec) < 2:
        return None
    if np.any(np.isnan(tuning_vec[:2])) or np.any(np.isinf(tuning_vec[:2])):
        return None
    x_min, x_max = 0.411397, 7.721101
    y_min, y_max = -6.013087, 7.707419
    restored_vec = tuning_vec.copy()
    restored_vec[0] = tuning_vec[0] * (x_max - x_min) + x_min
    restored_vec[1] = tuning_vec[1] * (y_max - y_min) + y_min
    return restored_vec


def scaleColor(tuning_vectors, absolute=True, bin=True, n_bins=N_BINS):
    """Map 2D tuning vectors to a scalar color value in [0, 1].

    2D vectors are reduced to a scalar via ``arctan2``, then
    normalized. With ``bin=True`` the normalized value is quantized to
    ``n_bins`` equal-width levels (default 10-bin); the returned scalar is the
    bin center-ish level in [0, 1].
    """
    tuning_scalars = []
    valid_indices = []
    for i, v in enumerate(tuning_vectors):
        if (v is not None and isinstance(v, np.ndarray)
                and v.ndim > 0 and len(v) >= 2
                and not np.any(np.isnan(v[:2])) and not np.any(np.isinf(v[:2]))):
            tuning_scalars.append(np.arctan2(v[1], v[0]))
            valid_indices.append(i)

    if not valid_indices:
        return np.full(len(tuning_vectors), 0.5)

    tuning_scalars = np.array(tuning_scalars)
    adjTuning = np.abs(tuning_scalars) if absolute else tuning_scalars

    c_max = np.max(adjTuning)
    c_min = np.min(adjTuning)
    c_range = c_max - c_min

    colors = np.full(len(tuning_vectors), 0.5)

    if c_range < 1e-9:
        normalized = np.zeros_like(adjTuning)
    else:
        normalized = (adjTuning - c_min) / c_range

    if bin:
        # 10 equal-width bins over [0, 1]; scalar = level in [0, 1].
        bin_idx = np.clip(np.floor(normalized * n_bins), 0, n_bins - 1)
        binned = bin_idx / (n_bins - 1)
        for i, idx in enumerate(valid_indices):
            colors[idx] = binned[i]
    else:
        for i, idx in enumerate(valid_indices):
            colors[idx] = normalized[i]

    return colors


def calculate_node_colors(data_file, predicted_tuning_map,
                          weight_matrix=None, mode="baseline"):
    """Node colors: V1 binned from true tuning (10-bin), V2-V4 propagated
    through the (column-normalized) weight matrix, else colored from
    predicted tuning."""
    with open(data_file, "rb") as f:
        raw_data = pickle.load(f)

    V1_nodes, V2_nodes, V3_nodes, V4_nodes = {}, {}, {}, {}
    for node_id, entry in raw_data.items():
        node_id_str = str(node_id)
        area = entry.get("area")
        tuning = entry.get("tuning")
        vec = restore_tuning_range(np.array(tuning, dtype=float))
        if area == 1:
            V1_nodes[node_id_str] = vec
        elif area == 2:
            V2_nodes[node_id_str] = vec
        elif area == 3:
            V3_nodes[node_id_str] = vec
        elif area == 4:
            V4_nodes[node_id_str] = vec

    V1_node_ids = list(V1_nodes.keys())
    V1_tuning_vectors = [V1_nodes[nid] for nid in V1_node_ids]
    V1_colors = scaleColor(V1_tuning_vectors, absolute=True, bin=True)

    result_colors = {}
    for i, node_id in enumerate(V1_node_ids):
        if i < len(V1_colors):
            result_colors[node_id] = V1_colors[i]

    if weight_matrix is not None:
        W = weight_matrix.cpu().numpy() if hasattr(weight_matrix, "cpu") else weight_matrix
        V1_color_array = np.array([result_colors.get(nid, 0.5) for nid in V1_node_ids])
        area_node_ids = list(V2_nodes.keys()) + list(V3_nodes.keys()) + list(V4_nodes.keys())
        if len(area_node_ids) > 0 and W.shape[1] > W.shape[0]:
            W_V2_to_V4 = W[:, W.shape[0]:]
            col_sums = np.sum(W_V2_to_V4, axis=0)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            W_V2_to_V4_normalized = W_V2_to_V4 / col_sums[np.newaxis, :]
            VnColors = (W_V2_to_V4_normalized.T @ V1_color_array).flatten()
            for i, node_id in enumerate(area_node_ids):
                if i < len(VnColors):
                    result_colors[node_id] = VnColors[i]
    else:
        area_node_ids = list(V2_nodes.keys()) + list(V3_nodes.keys()) + list(V4_nodes.keys())
        area_tuning_vectors = []
        for node_id in area_node_ids:
            if node_id in predicted_tuning_map:
                area_tuning_vectors.append(predicted_tuning_map[node_id])
            elif node_id in V2_nodes:
                area_tuning_vectors.append(V2_nodes[node_id])
            elif node_id in V3_nodes:
                area_tuning_vectors.append(V3_nodes[node_id])
            elif node_id in V4_nodes:
                area_tuning_vectors.append(V4_nodes[node_id])
            else:
                area_tuning_vectors.append(None)
        area_colors = scaleColor(area_tuning_vectors, absolute=True, bin=False)
        for i, node_id in enumerate(area_node_ids):
            if i < len(area_colors):
                result_colors[node_id] = area_colors[i]

    return result_colors


def save_node_colors_csv(node_colors, output_path):
    """Save node colors to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node_ID", "Color_Value"])
        for node_id, color_value in node_colors.items():
            writer.writerow([node_id, f"{color_value:.6f}"])
    print(f"Node colors saved to: {output_path}")
