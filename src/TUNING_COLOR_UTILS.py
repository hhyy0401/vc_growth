#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
from matplotlib.colors import ListedColormap


def get_tuning_colormap():
    """
    Get the discrete colormap for tuning colors (0.1 unit steps, 11 distinct colors).
    This is the canonical colormap used across all visualization code.
    
    Returns:
        ListedColormap: Matplotlib colormap with 11 discrete colors from red to purple
    """
    # Original rainbow colormap (commented out - using custom 10-color palette instead)
    # import matplotlib.pyplot as plt
    # return plt.cm.rainbow
    
    # Custom 10-color palette (0.0 to 0.9 in 0.1 steps)
    # 9 colors provided + purple added = 10 colors total
    colors_list = [
        '#73141B',  # 0.0 - Red (dark red)
        '#fd4405',  # 0.1 - Red-Orange
        '#fe9800',  # 0.2 - Orange
        '#fdff00',  # 0.3 - Yellow
        '#08fe01',  # 0.4 - Yellow-Green
        '#33cd32',  # 0.5 - Green
        '#00fefe',  # 0.6 - Cyan
        '#0096ff',  # 0.7 - Sky Blue
        '#0143ff',  # 0.8 - Blue
        '#4B0082',  # 0.9 - Dark Purple (Indigo)
    ]
    return ListedColormap(colors_list)


def get_v2_colormap():
    """
    Get the paper-style v2 discrete colormap (10 distinct colors).
    Order: Red (#B11226) (0.0) -> ... -> Indigo-Purple (#2C1E4A) (0.9).
    
    Returns:
        ListedColormap: Matplotlib colormap with 10 discrete colors.
    """
    colors_list = [
        '#B11226',  # 0.0 - Red (Bottom)
        '#E4572E',  # 0.1
        '#F7B23B',  # 0.2
        '#F2E84A',  # 0.3
        '#B5D64A',  # 0.4
        '#4FBF6B',  # 0.5
        '#1FA3A3',  # 0.6
        '#2A6FBB',  # 0.7
        '#2F4B8A',  # 0.8
        '#2C1E4A',  # 0.9 - Indigo-Purple (Top)
    ]
    return ListedColormap(colors_list)


def _phi_from_tuning_coords(
    tuning_coords: np.ndarray,
    v1_mask: np.ndarray | None,
    tag: str | None,
    *,
    anchor_margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute:
      - phi: atan2(dy, dx) in radians
      - r  : sqrt(dx^2 + dy^2)
    using a V1-derived center/anchor, matching the same "anchored + y-normalized" convention
    used in tuning color utilities (y scaled to [-1, 1], anchor placed left for lh / right for rh).

    Args:
        tuning_coords: (N,2) tuning coordinates [x, y] (y assumed in [0,1] before scaling)
        v1_mask: (N,) boolean mask of V1 nodes used to derive cy/x_min/x_max
        tag: hemisphere tag; contains "rh" -> right anchor, else left anchor
        anchor_margin: margin multiplier relative to V1 x-range

    Returns:
        (phi, r) where each is shape (N,)
    """
    tuning_coords = np.asarray(tuning_coords, dtype=float)
    if v1_mask is None:
        v1_mask = np.ones(tuning_coords.shape[0], dtype=bool)
    v1_mask = np.asarray(v1_mask, dtype=bool)
    if tuning_coords.ndim != 2 or tuning_coords.shape[1] != 2:
        raise ValueError(f"tuning_coords must be shape (N,2), got {tuning_coords.shape}")
    if v1_mask.shape != (tuning_coords.shape[0],):
        raise ValueError(f"v1_mask must be shape (N,), got {v1_mask.shape}")

    # Frame-aware normalization: rescale each axis to [0,1] using the V1 range so
    # that phi/r are invariant to the storage frame (normalized [0,1] vs native
    # visual degrees). Without this, the downstream y*2-1 convention assumes [0,1]
    # and distorts the phase profile when the data is in native-degree units.
    _v1n = tuning_coords if (not np.any(v1_mask)) else tuning_coords[v1_mask]
    _lo = _v1n.min(axis=0)
    _hi = _v1n.max(axis=0)
    _rng = np.where(_hi > _lo, _hi - _lo, 1.0)
    tuning_coords = (tuning_coords - _lo) / _rng

    # Match the exact anchor/cy convention used by compute_tuning_colors* utilities.
    v1_tuning_coords = tuning_coords if (not np.any(v1_mask)) else tuning_coords[v1_mask]
    ax_x, x_min, _x_max, cy = _anchor_and_cy_from_v1(v1_tuning_coords, tag=tag, anchor_margin=anchor_margin)

    xs = tuning_coords[:, 0]
    ys = tuning_coords[:, 1] * 2.0 - 1.0
    if ax_x < x_min:
        dx = np.maximum(xs - ax_x, 1e-12)
    else:
        dx = np.maximum(ax_x - xs, 1e-12)
    dy = ys - cy

    phi = np.arctan2(dy, dx)
    r = np.sqrt(dx * dx + dy * dy)
    return phi, r


def _anchor_and_cy_from_v1(v1_tuning_coords: np.ndarray, tag: str | None, *, anchor_margin: float = 0.05):
    """
    Shared parameter extraction for 1D color mappings:
    - cy: robust center on y (after scaling to [-1, 1])
    - ax_x: anchor x placed outside data range (side depends on tag)
    - x_min/x_max: for dx direction logic
    """
    v1_tuning_coords = np.asarray(v1_tuning_coords, dtype=float)
    if v1_tuning_coords.ndim != 2 or v1_tuning_coords.shape[1] != 2:
        raise ValueError(f"v1_tuning_coords must be shape (M, 2), got {v1_tuning_coords.shape}")

    xs_v1 = v1_tuning_coords[:, 0]
    ys_v1 = v1_tuning_coords[:, 1] * 2.0 - 1.0  # Scale y to [-1, 1]
    cy = float(np.median(ys_v1))

    x_min, x_max = float(np.min(xs_v1)), float(np.max(xs_v1))
    anchor_margin = float(anchor_margin)

    # Compute initial colors to find anchor point (mirrors compute_tuning_colors)
    is_rh = (tag is not None) and ("rh" in str(tag).lower())
    if is_rh:
        temp_ax_x = x_max + anchor_margin * (x_max - x_min)
    else:
        temp_ax_x = x_min - anchor_margin * (x_max - x_min)

    if is_rh:
        temp_dx_v1 = np.maximum(temp_ax_x - xs_v1, 1e-12)
    else:
        temp_dx_v1 = np.maximum(xs_v1 - temp_ax_x, 1e-12)

    temp_phi_v1 = np.arctan2(ys_v1 - cy, temp_dx_v1)
    temp_abs_phi_v1 = temp_phi_v1  # Use signed phi instead of abs

    temp_pmin = float(np.quantile(temp_abs_phi_v1, 0.01))
    temp_pmax = float(np.quantile(temp_abs_phi_v1, 0.99))
    try:
        coverage = float(os.getenv("COLOR_PHI_COVERAGE", "0.85"))
    except Exception:
        coverage = 0.85
    split_q = min(coverage, 1.0)
    temp_p_split = float(np.quantile(temp_abs_phi_v1, split_q))

    temp_colors = np.zeros_like(temp_abs_phi_v1, dtype=float)
    below_min_mask = temp_abs_phi_v1 < temp_pmin
    temp_colors[below_min_mask] = 0.0
    lower_mask = (temp_abs_phi_v1 >= temp_pmin) & (temp_abs_phi_v1 <= temp_p_split)
    if np.any(lower_mask):
        denom_lower = (temp_p_split - temp_pmin + 1e-12)
        temp_colors[lower_mask] = (temp_abs_phi_v1[lower_mask] - temp_pmin) / denom_lower * 0.8
    upper_mask = (temp_abs_phi_v1 > temp_p_split) & (temp_abs_phi_v1 <= temp_pmax)
    if np.any(upper_mask):
        denom_upper = (temp_pmax - temp_p_split + 1e-12)
        temp_colors[upper_mask] = 0.8 + (temp_abs_phi_v1[upper_mask] - temp_p_split) / denom_upper * 0.2
    above_max_mask = temp_abs_phi_v1 > temp_pmax
    temp_colors[above_max_mask] = 1.0

    low_color_mask = temp_colors <= 0.05
    if np.any(low_color_mask):
        low_color_xs = xs_v1[low_color_mask]
        if is_rh:
            rightmost_x = float(np.max(low_color_xs))
            ax_x = rightmost_x + anchor_margin * (x_max - x_min)
        else:
            leftmost_x = float(np.min(low_color_xs))
            ax_x = leftmost_x - anchor_margin * (x_max - x_min)
    else:
        if is_rh:
            ax_x = x_max + anchor_margin * (x_max - x_min)
        else:
            ax_x = x_min - anchor_margin * (x_max - x_min)

    if (not is_rh) and ax_x >= x_min:
        ax_x = x_min - anchor_margin * (x_max - x_min)
    elif is_rh and ax_x <= x_max:
        ax_x = x_max + anchor_margin * (x_max - x_min)

    return ax_x, x_min, x_max, cy


def compute_tuning_colors_r(tuning_coords, v1_mask=None, tag=None):
    """
    Compute color values from tuning coordinates using *radial distance* r instead of arctan2(phi).

    Mapping:
    - r_max is robust: 99th percentile of V1 r (top 1% treated as outliers and clipped).
    - After clipping to [0, r_max], we assign **10 bins by node-count (deciles)**:
        * smallest r group -> 0.9 (blue-ish end)
        * largest r group  -> 0.0 (red end)

    Returns:
        Array of shape (N,) with values in [0.0, 0.9].
    """
    if tuning_coords is None:
        raise ValueError("tuning_coords must not be None")
    tuning_coords = np.asarray(tuning_coords, dtype=float)
    if tuning_coords.size == 0:
        return np.array([], dtype=float)
    if tuning_coords.ndim != 2 or tuning_coords.shape[1] != 2:
        raise ValueError(f"tuning_coords must be shape (N, 2), got {tuning_coords.shape}")

    N = tuning_coords.shape[0]

    if v1_mask is None:
        v1_mask = np.ones(N, dtype=bool)
    v1_mask = np.asarray(v1_mask, dtype=bool)
    if v1_mask.shape != (N,):
        raise ValueError(f"v1_mask must be shape ({N},), got {v1_mask.shape}")
    
    # Reuse the shared anchored/y-normalized geometry used elsewhere.
    _phi, r = _phi_from_tuning_coords(tuning_coords, v1_mask=v1_mask, tag=tag)
    r_v1 = r if (not np.any(v1_mask)) else r[v1_mask]
    r_max = float(np.quantile(r_v1, 0.99)) if r_v1.size > 0 else 0.0
    if not np.isfinite(r_max) or r_max <= 0.0:
        # Degenerate: everything maps to "blue" end
        return np.full((N,), 0.9, dtype=float)

    r_clip = np.clip(r, 0.0, r_max)  # clip top 1% outliers to r_max

    # Node-count based 10-bin assignment (deciles)
    # Sort V1 r values only to find quantile boundaries
    sorted_r_v1 = np.sort(np.clip(r_v1, 0.0, r_max))

    n_v1 = len(sorted_r_v1)
    if n_v1 == 0:
        return np.full((N,), 0.9, dtype=float)

    group_size = n_v1 / 10.0
    quantile_indices = [int(i * group_size) for i in range(11)]
    quantile_indices[-1] = n_v1 - 1
    quantile_values = sorted_r_v1[quantile_indices]

    colors = np.zeros_like(r_clip, dtype=float)
    for i in range(10):
        if i == 0:
            mask = (r_clip >= quantile_values[i]) & (r_clip <= quantile_values[i + 1])
        else:
            mask = (r_clip > quantile_values[i]) & (r_clip <= quantile_values[i + 1])

        # smallest r -> 0.9, largest r -> 0.0
        colors[mask] = (9 - i) / 10.0

    return np.clip(colors, 0.0, 0.9)


def compute_tuning_colors_r_v2(tuning_coords, v1_mask=None, tag=None):
    """
    Compute color values from tuning coordinates using radial distance r for v2 (9 bins).
    Returns values in [0.0, 0.8].
    """
    if tuning_coords is None: raise ValueError("tuning_coords must not be None")
    tuning_coords = np.asarray(tuning_coords, dtype=float)
    if tuning_coords.size == 0: return np.array([], dtype=float)
    N = tuning_coords.shape[0]
    if v1_mask is None: v1_mask = np.ones(N, dtype=bool)
    v1_mask = np.asarray(v1_mask, dtype=bool)
    
    _phi, r = _phi_from_tuning_coords(tuning_coords, v1_mask=v1_mask, tag=tag)
    r_v1 = r if (not np.any(v1_mask)) else r[v1_mask]
    r_max = float(np.quantile(r_v1, 0.99)) if r_v1.size > 0 else 0.0
    if not np.isfinite(r_max) or r_max <= 0.0: return np.full((N,), 0.8, dtype=float)
    r_clip = np.clip(r, 0.0, r_max)
    sorted_r_v1 = np.sort(np.clip(r_v1, 0.0, r_max))
    n_v1 = len(sorted_r_v1)
    if n_v1 == 0: return np.full((N,), 0.8, dtype=float)

    group_size = n_v1 / 10.0
    quantile_indices = [int(i * group_size) for i in range(11)]
    quantile_indices[-1] = n_v1 - 1
    quantile_values = sorted_r_v1[quantile_indices]

    colors = np.zeros_like(r_clip, dtype=float)
    for i in range(10):
        if i == 0: mask = (r_clip >= quantile_values[i]) & (r_clip <= quantile_values[i + 1])
        else: mask = (r_clip > quantile_values[i]) & (r_clip <= quantile_values[i + 1])
        # scale to 0.0, 0.1, ..., 0.9
        colors[mask] = (9 - i) / 10.0
    return np.clip(colors, 0.0, 0.9)


def compute_tuning_colors(tuning_coords, v1_mask=None, tag=None):
    """
    Compute color values from tuning coordinates using V1-based color map.
    
    Args:
        tuning_coords: Array of shape (N, 2) with tuning coordinates [x, y]
        v1_mask: Optional boolean array of shape (N,) indicating V1 nodes (area==1)
                 If None, uses all nodes to compute color map parameters
        tag: Optional tag string (e.g., "lh", "rh") to determine anchor side
             "rh" -> right anchor, "lh" or None -> left anchor
    
    Returns:
        Array of shape (N,) with normalized color values in [0, 1] range
    """
    if tuning_coords.size == 0:
        return np.array([], dtype=float)
    
    tuning_coords = np.asarray(tuning_coords, dtype=float)
    if tuning_coords.ndim != 2 or tuning_coords.shape[1] != 2:
        raise ValueError(f"tuning_coords must be shape (N, 2), got {tuning_coords.shape}")
    
    N = tuning_coords.shape[0]
    
    # Determine V1 mask
    if v1_mask is None:
        v1_mask = np.ones(N, dtype=bool)
    
    v1_mask = np.asarray(v1_mask, dtype=bool)
    if v1_mask.shape != (N,):
        raise ValueError(f"v1_mask must be shape ({N},), got {v1_mask.shape}")

    # Frame-aware: normalize each axis to [0,1] over V1 so polar colors are invariant
    # to the storage frame (normalized [0,1] vs native visual degrees).
    _v1n = tuning_coords if (not np.any(v1_mask)) else tuning_coords[v1_mask]
    _lo = _v1n.min(axis=0)
    _hi = _v1n.max(axis=0)
    _rng = np.where(_hi > _lo, _hi - _lo, 1.0)
    tuning_coords = (tuning_coords - _lo) / _rng

    # Extract V1 tuning coordinates from tuning_coords
    if not np.any(v1_mask):
        v1_tuning_coords = tuning_coords
    else:
        v1_tuning_coords = tuning_coords[v1_mask]

    # Step 1: Compute anchor/cy from V1 tuning (shared helper; same convention as compute_tuning_colors_r)
    ax_x, x_min, _x_max, cy = _anchor_and_cy_from_v1(v1_tuning_coords, tag=tag)

    xs_v1 = v1_tuning_coords[:, 0]
    ys_v1 = v1_tuning_coords[:, 1] * 2.0 - 1.0  # Scale y to [-1, 1]
    if ax_x < x_min:
        dx_v1 = np.maximum(xs_v1 - ax_x, 1e-12)
    else:
        dx_v1 = np.maximum(ax_x - xs_v1, 1e-12)

    phi_v1 = np.arctan2(ys_v1 - cy, dx_v1)
    abs_phi_v1 = phi_v1
    pmin = float(np.quantile(abs_phi_v1, 0.01))
    pmax = float(np.quantile(abs_phi_v1, 0.99))
    
    # Step 2: Apply same color map parameters to all tuning coordinates
    xs = tuning_coords[:, 0]
    ys = tuning_coords[:, 1] * 2.0 - 1.0
    
    if ax_x < x_min:
        dx = np.maximum(xs - ax_x, 1e-12)
    else:
        dx = np.maximum(ax_x - xs, 1e-12)
    
    phi = np.arctan2(ys - cy, dx)
    abs_phi = phi
    
    # Step 3: Apply quantile-based mapping (node count-based distribution)
    # Sort V1 phi values only to find quantile boundaries
    abs_phi_v1_clipped = np.clip(abs_phi_v1, pmin, pmax)
    sorted_phi_v1 = np.sort(abs_phi_v1_clipped)

    n_v1 = len(sorted_phi_v1)
    if n_v1 == 0:
        return np.zeros_like(abs_phi, dtype=float)
    
    group_size = n_v1 / 10.0
    quantile_indices = [int(i * group_size) for i in range(11)]
    quantile_indices[-1] = n_v1 - 1
    quantile_values = sorted_phi_v1[quantile_indices]
    
    colors = np.zeros_like(abs_phi, dtype=float)
    abs_phi_clipped = np.clip(abs_phi, pmin, pmax)
    for i in range(10):
        if i == 0:
            mask = (abs_phi_clipped >= quantile_values[i]) & (abs_phi_clipped <= quantile_values[i+1])
        else:
            mask = (abs_phi_clipped > quantile_values[i]) & (abs_phi_clipped <= quantile_values[i+1])
        
        colors[mask] = i / 10.0
    
    return colors


def compute_tuning_colors_v2(tuning_coords, v1_mask=None, tag=None):
    """
    Compute color values from tuning coordinates using V1-based mapping for v2 (9 bins).
    Returns values in [0.0, 0.8] range.
    """
    if tuning_coords.size == 0: return np.array([], dtype=float)
    tuning_coords = np.asarray(tuning_coords, dtype=float)
    N = tuning_coords.shape[0]
    if v1_mask is None: v1_mask = np.ones(N, dtype=bool)
    v1_mask = np.asarray(v1_mask, dtype=bool)
    v1_tuning_coords = tuning_coords if not np.any(v1_mask) else tuning_coords[v1_mask]
    
    ax_x, x_min, _x_max, cy = _anchor_and_cy_from_v1(v1_tuning_coords, tag=tag)
    xs_v1 = v1_tuning_coords[:, 0]
    ys_v1 = v1_tuning_coords[:, 1] * 2.0 - 1.0
    dx_v1 = np.maximum(xs_v1 - ax_x, 1e-12) if ax_x < x_min else np.maximum(ax_x - xs_v1, 1e-12)
    phi_v1 = np.arctan2(ys_v1 - cy, dx_v1)
    pmin, pmax = float(np.quantile(phi_v1, 0.01)), float(np.quantile(phi_v1, 0.99))
    
    xs = tuning_coords[:, 0]
    ys = tuning_coords[:, 1] * 2.0 - 1.0
    dx = np.maximum(xs - ax_x, 1e-12) if ax_x < x_min else np.maximum(ax_x - xs, 1e-12)
    phi = np.arctan2(ys - cy, dx)
    phi_clipped = np.clip(phi, pmin, pmax)
    sorted_phi_v1 = np.sort(np.clip(phi_v1, pmin, pmax))
    n_v1 = len(sorted_phi_v1)
    if n_v1 == 0: return np.zeros_like(phi, dtype=float)
    
    group_size = n_v1 / 10.0
    quantile_indices = [int(i * group_size) for i in range(11)]
    quantile_indices[-1] = n_v1 - 1
    quantile_values = sorted_phi_v1[quantile_indices]
    
    colors = np.zeros_like(phi, dtype=float)
    for i in range(10):
        if i == 0: mask = (phi_clipped >= quantile_values[i]) & (phi_clipped <= quantile_values[i+1])
        else: mask = (phi_clipped > quantile_values[i]) & (phi_clipped <= quantile_values[i+1])
        # scale to 0.0, 0.1, ..., 0.9
        colors[mask] = i / 10.0
    return colors


# -----------------------------------------------------------------------------
# Reusable utilities for "interpolated" / dense plotting (plot_boundary-style)
# -----------------------------------------------------------------------------

def round_color_bins(colors: np.ndarray) -> np.ndarray:
    """
    Round scalar colors to the discrete palette bins (0.0, 0.1, ..., 0.9).
    """
    c = np.asarray(colors, dtype=float)
    c = np.round(c * 10.0) / 10.0
    return np.clip(c, 0.0, 0.9)


def should_flip_y_red_bottom(xy: np.ndarray, colors: np.ndarray, q: float = 0.10) -> bool:
    """
    Decide whether to flip across x-axis (y -> -y) so that:
      bottom (low y) is 'redder' (smaller color) and top (high y) is 'bluer' (larger color).
    Uses robust quantile averages at y<=q and y>=1-q.
    """
    xy = np.asarray(xy, dtype=float)
    c = np.asarray(colors, dtype=float)
    if xy.size == 0 or c.size == 0 or xy.shape[0] != c.shape[0]:
        return False
    if xy.shape[0] < 20:
        return False
    y = xy[:, 1]
    q = float(q)
    lo = y <= np.quantile(y, q)
    hi = y >= np.quantile(y, 1.0 - q)
    if not np.any(lo) or not np.any(hi):
        return False
    bottom_mean = float(np.mean(c[lo]))
    top_mean = float(np.mean(c[hi]))
    return bottom_mean > top_mean


def remove_outliers_xyc(xy: np.ndarray, colors: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    IsolationForest over (x, y, color) to drop gross outliers.
    Returns boolean mask of inliers.
    """
    from sklearn.ensemble import IsolationForest

    xy = np.asarray(xy, dtype=float)
    c = np.asarray(colors, dtype=float).reshape(-1, 1)
    X = np.hstack([xy, c])
    iso = IsolationForest(contamination=float(contamination), random_state=42)
    return (iso.fit_predict(X) == 1)


def keep_largest_island(xy: np.ndarray, eps: float = 3.0, min_samples: int = 10) -> np.ndarray:
    """
    Keep only the largest DBSCAN cluster (excluding noise).
    Returns boolean mask of kept points.
    """
    from sklearn.cluster import DBSCAN

    xy = np.asarray(xy, dtype=float)
    clustering = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    labels = clustering.fit_predict(xy)
    mask = labels != -1
    if not np.any(mask):
        return np.ones(len(xy), dtype=bool)
    unique, counts = np.unique(labels[mask], return_counts=True)
    keep_label = unique[int(np.argmax(counts))]
    return labels == keep_label


def generate_uniform_points_fixed_spacing(
    xy: np.ndarray,
    spacing: float = 0.75,
    epsilon: float = 1.0,
    pad_frac: float = 0.10,
) -> np.ndarray:
    """
    Axis-aligned uniform grid with fixed spacing; keep points within epsilon of original points.
    Fills enclosed holes (binary_fill_holes) to get a solid region.
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import binary_fill_holes

    xy = np.asarray(xy, dtype=float)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xpad = (xmax - xmin) * float(pad_frac)
    ypad = (ymax - ymin) * float(pad_frac)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    spacing = float(spacing)
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    Xg, Yg = np.meshgrid(xs, ys)
    candidates = np.column_stack([Xg.ravel(), Yg.ravel()])

    tree = cKDTree(xy)
    d, _ = tree.query(candidates, k=1)
    valid = (d <= float(epsilon))

    grid_mask = valid.reshape(len(ys), len(xs))
    filled = binary_fill_holes(grid_mask)
    keep = filled.ravel()
    return candidates[keep]


def gpr_interpolate_1d(xy_train: np.ndarray, y_train: np.ndarray, xy_query: np.ndarray) -> np.ndarray:
    """
    GPR interpolate a scalar field onto query points.
    Kernel matches plot_boundary-style (Matern + WhiteKernel).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import WhiteKernel, Matern

    xy_train = np.asarray(xy_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    xy_query = np.asarray(xy_query, dtype=float)
    kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gpr.fit(xy_train, y_train)
    return gpr.predict(xy_query)


def interpolate_colors_on_uniform_grid(
    xy: np.ndarray,
    colors: np.ndarray,
    *,
    spacing: float = 0.75,
    epsilon: float = 1.0,
    contamination: float = 0.05,
    dbscan_eps: float = 3.0,
    dbscan_min_samples: int = 10,
    pad_frac: float = 0.10,
    clip_min: float = 0.0,
    clip_max: float = 0.9,
    round_bins: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot-boundary style dense interpolation pipeline for scalar tuning colors.

    Steps:
    - outlier removal (IsolationForest) on (x,y,color)
    - island removal (DBSCAN keep largest cluster)
    - generate uniform points on an axis-aligned grid (fixed spacing) within epsilon of data, with hole filling
    - GPR interpolate scalar colors onto the uniform points
    - clip + optionally round to discrete bins

    Returns:
      (xy_uniform, colors_uniform)
    """
    xy = np.asarray(xy, dtype=float)
    c = np.asarray(colors, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N,2), got {xy.shape}")
    if c.shape[0] != xy.shape[0]:
        raise ValueError(f"colors must be (N,), got {c.shape} vs xy {xy.shape}")

    inlier = remove_outliers_xyc(xy, c, contamination=float(contamination))
    xy_c = xy[inlier]
    c_c = c[inlier]

    island = keep_largest_island(xy_c, eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
    xy_c = xy_c[island]
    c_c = c_c[island]

    xy_u = generate_uniform_points_fixed_spacing(
        xy_c, spacing=float(spacing), epsilon=float(epsilon), pad_frac=float(pad_frac)
    )
    c_u = gpr_interpolate_1d(xy_c, c_c, xy_u)
    c_u = np.clip(c_u, float(clip_min), float(clip_max))
    if round_bins:
        c_u = round_color_bins(c_u)
    return xy_u, c_u


def rasterize_uniform_grid(
    xy_uniform: np.ndarray,
    values: np.ndarray,
    *,
    spacing: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Rasterize points on a fixed-spacing axis-aligned grid into a 2D image array.

    Assumes `xy_uniform` was generated by `generate_uniform_points_fixed_spacing` with the same spacing.

    Returns:
      (img, extent) where img is (H,W) with NaNs for empty cells, and extent is (xmin, xmax, ymin, ymax)
      suitable for `imshow(..., extent=extent, origin='lower', interpolation='nearest')`.
    """
    xy = np.asarray(xy_uniform, dtype=float)
    v = np.asarray(values, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy_uniform must be (N,2), got {xy.shape}")
    if v.shape[0] != xy.shape[0]:
        raise ValueError(f"values must be (N,), got {v.shape} vs xy {xy.shape}")

    s = float(spacing)
    # Snap to grid indices
    xmin = float(np.min(xy[:, 0]))
    ymin = float(np.min(xy[:, 1]))
    ix = np.rint((xy[:, 0] - xmin) / s).astype(int)
    iy = np.rint((xy[:, 1] - ymin) / s).astype(int)
    w = int(ix.max()) + 1
    h = int(iy.max()) + 1

    img = np.full((h, w), np.nan, dtype=float)
    img[iy, ix] = v

    half = s / 2.0
    extent = (xmin - half, xmin + (w - 1) * s + half, ymin - half, ymin + (h - 1) * s + half)
    return img, extent


# ---------------------------------------------------------------------------
# Visual-degree denormalization
# ---------------------------------------------------------------------------
# Denormalization parameters: convert normalized tuning [0,1] back to visual
# degrees (X, Y in the visual field).
# Format per entry: (scale_x, offset_x, scale_y, offset_y)
# Formula:  tuning_vd = tuning_norm * scale + offset
# Derived from *_original_lh.pkl  (tuning_original field)
#        and  *_rh.pkl             (tuning_original field).
_TUNING_DENORM_PARAMS = {
    'R1': {
        'lh': (1.0, 0.0, 1.0, 0.0),  # R1 lh migrated to native visual degrees (was 10.1010,-2.3799,13.8060,-5.2969)
        'rh': (1.0, 0.0, 1.0, 0.0),  # R1 rh already in visual degrees
    },
    # S1-S6 migrated to native visual degrees (all cached tsvs denormalized in
    # place), so denorm is now identity. OLD [0,1]->deg params kept for reference:
    #   S1 lh (12.2134,-10.5000,15.1515,-6.2300)  rh (14.9360,-2.1200,18.1140,-7.8370)
    #   S2 lh (12.7479,-2.8650,27.9710,-14.6000)  rh (14.5423,-13.9600,14.9155,-6.1580)
    #   S3 lh (10.3792,0.0008,15.1021,-5.8250)    rh (12.3051,-10.1500,19.6700,-9.4080)
    #   S4 lh (10.7626,0.0134,20.0380,-7.9520)    rh (11.7734,-11.3100,14.2716,-6.9610)
    #   S5 lh (15.2130,-2.6710,21.5858,-11.6900)  rh (17.8800,-17.8800,19.0959,-10.8200)
    #   S6 lh (12.5860,-2.0070,17.5232,-10.3500)  rh (13.3018,-10.7100,17.5117,-8.0590)
    'S1': {'lh': (1.0, 0.0, 1.0, 0.0), 'rh': (1.0, 0.0, 1.0, 0.0)},
    'S2': {'lh': (1.0, 0.0, 1.0, 0.0), 'rh': (1.0, 0.0, 1.0, 0.0)},
    'S3': {'lh': (1.0, 0.0, 1.0, 0.0), 'rh': (1.0, 0.0, 1.0, 0.0)},
    'S4': {'lh': (1.0, 0.0, 1.0, 0.0), 'rh': (1.0, 0.0, 1.0, 0.0)},
    'S5': {'lh': (1.0, 0.0, 1.0, 0.0), 'rh': (1.0, 0.0, 1.0, 0.0)},
    'S6': {'lh': (1.0, 0.0, 1.0, 0.0), 'rh': (1.0, 0.0, 1.0, 0.0)},
}


def get_tuning_denorm_params(data_name: str, tag: str):
    """Return (scale_x, offset_x, scale_y, offset_y) for a subject/hemi.

    *data_name* can be e.g. ``"R1_gpr_grid"`` or ``"S1_S2_gpr_grid"``; the
    subject is taken from the first token before ``'_'``.
    """
    subject = data_name.split('_')[0]
    tag_key = tag.lower()
    if subject in _TUNING_DENORM_PARAMS and tag_key in _TUNING_DENORM_PARAMS[subject]:
        return _TUNING_DENORM_PARAMS[subject][tag_key]
    return (1.0, 0.0, 1.0, 0.0)


def tuning_to_visual_degrees(tuning, data_name: str, tag: str):
    """Convert tuning array (N,2) to visual-degree coordinates."""
    sx, ox, sy, oy = get_tuning_denorm_params(data_name, tag)
    tuning = np.asarray(tuning, dtype=float)
    out = np.empty_like(tuning)
    out[:, 0] = tuning[:, 0] * sx + ox
    out[:, 1] = tuning[:, 1] * sy + oy
    return out


def euclidean_error_visual_degrees(true_tuning, pred_tuning, data_name: str, tag: str):
    """Per-node Euclidean distance in visual degrees.

    Returns 1-D array of shape (N,).
    """
    true_vd = tuning_to_visual_degrees(true_tuning, data_name, tag)
    pred_vd = tuning_to_visual_degrees(pred_tuning, data_name, tag)
    return np.sqrt(np.sum((true_vd - pred_vd) ** 2, axis=1))
