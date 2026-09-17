#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
from matplotlib.colors import ListedColormap


def get_tuning_colormap():
    """Discrete colormap used for every map: 10 bins from red to purple."""
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


def _phi_from_tuning_coords(
    tuning_coords: np.ndarray,
    v1_mask: np.ndarray | None,
    tag: str | None,
    *,
    anchor_margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Polar coordinates of tuning vectors about the V1 anchor.

    In: (N, 2) tuning vectors in visual degrees, a V1 mask, and the hemisphere
    tag. Out: (phi, r), the angle in radians and the eccentricity in visual
    degrees, both measured from the anchor that the V1 tuning defines.
    """
    tuning_coords = np.asarray(tuning_coords, dtype=float)
    if v1_mask is None:
        v1_mask = np.ones(tuning_coords.shape[0], dtype=bool)
    v1_mask = np.asarray(v1_mask, dtype=bool)
    if tuning_coords.ndim != 2 or tuning_coords.shape[1] != 2:
        raise ValueError(f"tuning_coords must be shape (N,2), got {tuning_coords.shape}")
    if v1_mask.shape != (tuning_coords.shape[0],):
        raise ValueError(f"v1_mask must be shape (N,), got {v1_mask.shape}")

    # The anchor is derived on axes rescaled to [0, 1] over the V1 range, then
    # phi and r are converted back to visual degrees.
    frame = tuning_coords if not np.any(v1_mask) else tuning_coords[v1_mask]
    lo, hi = frame.min(axis=0), frame.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    scaled = (tuning_coords - lo) / rng

    v1_scaled = scaled if not np.any(v1_mask) else scaled[v1_mask]
    ax_x, x_min, _x_max, cy = _anchor_and_cy_from_v1(v1_scaled, tag=tag, anchor_margin=anchor_margin)

    xs = scaled[:, 0]
    ys = scaled[:, 1] * 2.0 - 1.0
    dx = np.maximum(xs - ax_x, 1e-12) if ax_x < x_min else np.maximum(ax_x - xs, 1e-12)
    dy = ys - cy
    dx = dx * rng[0]
    dy = dy * (rng[1] / 2.0)
    return np.arctan2(dy, dx), np.sqrt(dx * dx + dy * dy)


def _anchor_and_cy_from_v1(v1_tuning_coords: np.ndarray, tag: str | None, *, anchor_margin: float = 0.05):
    """The V1 anchor of one hemisphere.

    In: (M, 2) V1 tuning vectors and the hemisphere tag. Out: the anchor x, the
    V1 x range, and the anchor y, on axes scaled to [-1, 1]. The anchor sits just
    outside the V1 range, on the left for lh and on the right for rh.
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
    """Eccentricity colors: r about the V1 anchor, binned into deciles of the V1
    distribution, from 0.9 at the fovea to 0.0 at the periphery."""
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

    # Nodes nearer the fovea than every V1 node fall below the first decile edge
    # and would otherwise keep the initial 0.0 (the peripheral colour).
    colors[r_clip < quantile_values[0]] = 0.9

    return np.clip(colors, 0.0, 0.9)


def compute_tuning_colors(tuning_coords, v1_mask=None, tag=None):
    """Polar-angle colors: phi about the V1 anchor, binned so that the bins are
    defined by the V1 nodes and then applied to every node. Out: (N,) in [0, 1]."""
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


# -----------------------------------------------------------------------------
# Reusable utilities for "interpolated" / dense plotting (plot_boundary-style)
# -----------------------------------------------------------------------------

def round_color_bins(colors: np.ndarray) -> np.ndarray:
    """Round colors to the palette bins 0.0, 0.1, ..., 0.9."""
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


def tuning_to_polar(tuning, v1_tuning_coords, tag: str | None):
    """Eccentricity and polar angle of *tuning*, about the V1 anchor.

    *tuning* and *v1_tuning_coords* are (N, 2) arrays of tuning vectors in
    visual degrees, as stored in the input pkl and written to the prediction
    TSV. The polar frame is the one used throughout the paper: the pole is the
    V1 anchor that :func:`_anchor_and_cy_from_v1` derives from the V1 tuning of
    that hemisphere, and the angle increases toward the upper visual field, so
    the two hemispheres are mirror images of each other.

    Returns ``(eccentricity, polar_angle)``, the first in visual degrees and the
    second in degrees within (-180, 180].
    """
    tuning = np.asarray(tuning, dtype=float)
    ax_x, _x_min, _x_max, cy = _anchor_and_cy_from_v1(v1_tuning_coords, tag=tag)
    c_y = (cy + 1.0) / 2.0                       # undo the y scaling to [-1, 1]
    is_rh = (tag is not None) and ("rh" in str(tag).lower())
    dx = (ax_x - tuning[:, 0]) if is_rh else (tuning[:, 0] - ax_x)
    dy = tuning[:, 1] - c_y
    ecc = np.hypot(tuning[:, 0] - ax_x, dy)
    return ecc, np.degrees(np.arctan2(dy, dx))


