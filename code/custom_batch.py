"""
Custom batch mode: spatial-geometry-based batching using V1 boundary lines.

Naming: custom_{mode}_{eccOrder}
  mode ∈ {angle, polar, euclidean, x}
  eccOrder ∈ {up, down, random}  (eccentricity ordering within each batch)

Usage:
  from custom_batch import get_custom_node_order
  ordered_nodes, batch_assignments = get_custom_node_order(DF, "angle", "up", n_batches=30)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_v1_boundary(DF):
    """Detect V1/V2 boundary and split into two lines emanating from is_center."""
    coords = DF[["x", "y"]].values.astype(float)
    areas = DF["area"].values.astype(int)

    nn = NearestNeighbors(n_neighbors=4, algorithm="ball_tree")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    boundary_mask = np.zeros(len(DF), dtype=bool)
    for i in range(len(DF)):
        neighbor_areas = set(areas[indices[i, 1:]])
        if 1 in neighbor_areas and 2 in neighbor_areas:
            boundary_mask[i] = True

    boundary_indices = np.where(boundary_mask)[0]
    if len(boundary_indices) == 0:
        raise ValueError("No V1 boundary nodes found! Check knn graph and area labels.")

    center_mask = DF["is_center"].values.astype(int) == 1
    if not np.any(center_mask):
        raise ValueError("No is_center node found in DF.")
    center_idx = int(np.where(center_mask)[0][0])
    center_xy = coords[center_idx]

    bnd_coords = coords[boundary_indices]
    angles_from_center = np.arctan2(
        bnd_coords[:, 1] - center_xy[1],
        bnd_coords[:, 0] - center_xy[0]
    )

    sorted_order = np.argsort(angles_from_center)
    sorted_angles = angles_from_center[sorted_order]
    sorted_bnd_indices = boundary_indices[sorted_order]

    diffs = np.diff(sorted_angles)
    wrap_diff = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
    all_diffs = np.append(diffs, wrap_diff)
    gap1_idx = np.argmax(all_diffs)

    all_diffs_copy = all_diffs.copy()
    all_diffs_copy[gap1_idx] = -np.inf
    gap2_idx = np.argmax(all_diffs_copy)

    gaps = sorted([gap1_idx, gap2_idx])
    line1_indices = sorted_bnd_indices[gaps[0] + 1: gaps[1] + 1]
    line2_indices = np.concatenate([
        sorted_bnd_indices[gaps[1] + 1:],
        sorted_bnd_indices[:gaps[0] + 1]
    ])

    def order_line_from_center(line_idx):
        if len(line_idx) == 0:
            return line_idx
        dists = np.sqrt(np.sum((coords[line_idx] - center_xy) ** 2, axis=1))
        return line_idx[np.argsort(dists)]

    line1 = order_line_from_center(line1_indices)
    line2 = order_line_from_center(line2_indices)

    return line1, line2, center_xy, center_idx


def _boundary_line_coords(DF, line_indices):
    """Get (x, y) coordinates for a boundary line."""
    return DF.iloc[line_indices][["x", "y"]].values.astype(float)


def _boundary_midangle(DF, line1, line2, center_xy):
    """Midpoint angle (radians) between the two boundary line tips."""
    if len(line1) == 0 or len(line2) == 0:
        return 0.0
    coords = DF[["x", "y"]].values.astype(float)
    p1 = coords[line1[-1]]
    p2 = coords[line2[-1]]
    a1 = np.arctan2(p1[1] - center_xy[1], p1[0] - center_xy[0])
    a2 = np.arctan2(p2[1] - center_xy[1], p2[0] - center_xy[0])
    mid = np.arctan2(
        np.sin(a1) + np.sin(a2),
        np.cos(a1) + np.cos(a2)
    )
    return mid


def _distance_to_boundary(coords_2d, DF, line1, line2):
    """Minimum distance from each point to the nearest boundary line segment."""
    all_coords = DF[["x", "y"]].values.astype(float)

    def dist_to_line(pts, line_idx):
        """Distance from pts (N,2) to a polyline defined by line_idx."""
        if len(line_idx) < 2:
            if len(line_idx) == 1:
                return np.sqrt(np.sum((pts - all_coords[line_idx[0]]) ** 2, axis=1))
            return np.full(len(pts), np.inf)
        line_pts = all_coords[line_idx]
        min_d = np.full(len(pts), np.inf)
        for i in range(len(line_pts) - 1):
            a = line_pts[i]
            b = line_pts[i + 1]
            ab = b - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-12:
                d = np.sqrt(np.sum((pts - a) ** 2, axis=1))
            else:
                t = np.clip(((pts - a) @ ab) / ab_sq, 0, 1)
                proj = a + np.outer(t, ab)
                d = np.sqrt(np.sum((pts - proj) ** 2, axis=1))
            min_d = np.minimum(min_d, d)
        return min_d

    d1 = dist_to_line(coords_2d, line1)
    d2 = dist_to_line(coords_2d, line2)
    return np.minimum(d1, d2)


def _is_v1_side(DF, point_xy, line1, line2, center_xy):
    """Check if a point is on the V1 side of the boundary."""
    areas = DF["area"].values.astype(int)
    coords = DF[["x", "y"]].values.astype(float)
    v1_centroid = coords[areas == 1].mean(axis=0)

    mid_angle = _boundary_midangle(DF, line1, line2, center_xy)
    v1_dir = np.arctan2(v1_centroid[1] - center_xy[1], v1_centroid[0] - center_xy[0])
    pt_dir = np.arctan2(point_xy[1] - center_xy[1], point_xy[0] - center_xy[0])

    def _angle_diff(a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

    return np.abs(_angle_diff(pt_dir, v1_dir)) < np.abs(_angle_diff(mid_angle, v1_dir))


def _get_vn_mask(DF):
    """Return boolean mask for non-V1 nodes."""
    return DF["area"].values.astype(int) != 1


def compute_batches_angle(DF, line1, line2, center_xy, center_idx, n_batches=30):
    """Angle-based batching: divide the non-V1 angular range into equal wedges."""
    vn_mask = _get_vn_mask(DF)
    coords = DF[["x", "y"]].values.astype(float)
    areas = DF["area"].values.astype(int)
    vn_indices = np.where(vn_mask)[0]

    if len(vn_indices) == 0:
        return {}

    v1_centroid = coords[areas == 1].mean(axis=0)
    v1_dir = np.arctan2(v1_centroid[1] - center_xy[1],
                        v1_centroid[0] - center_xy[0])

    if len(line1) > 0:
        tip1 = np.arctan2(coords[line1[-1], 1] - center_xy[1],
                          coords[line1[-1], 0] - center_xy[0])
    else:
        tip1 = v1_dir + np.pi / 2
    if len(line2) > 0:
        tip2 = np.arctan2(coords[line2[-1], 1] - center_xy[1],
                          coords[line2[-1], 0] - center_xy[0])
    else:
        tip2 = v1_dir - np.pi / 2

    def _angle_in_arc_ccw(angle, arc_start, arc_end):
        a = (angle - arc_start) % (2 * np.pi)
        span = (arc_end - arc_start) % (2 * np.pi)
        return a <= span

    v1_in_arc_a = _angle_in_arc_ccw(v1_dir, tip1, tip2)

    if v1_in_arc_a:
        away_start = tip2
        away_end = tip1
    else:
        away_start = tip1
        away_end = tip2

    away_span = (away_end - away_start) % (2 * np.pi)
    if away_span < 1e-6:
        away_span = 2 * np.pi

    vn_coords = coords[vn_indices]
    vn_angles = np.arctan2(vn_coords[:, 1] - center_xy[1],
                           vn_coords[:, 0] - center_xy[0])
    vn_in_away = (vn_angles - away_start) % (2 * np.pi)

    bin_edges = np.linspace(0, away_span, n_batches + 1)

    batches = {}
    for b in range(n_batches):
        in_bin = (vn_in_away >= bin_edges[b]) & (vn_in_away < bin_edges[b + 1])
        batch_nodes = vn_indices[in_bin]
        if len(batch_nodes) > 0:
            batches[b] = batch_nodes.tolist()

    assigned = set()
    for nodes in batches.values():
        assigned.update(nodes)
    unassigned = [idx for idx in vn_indices if idx not in assigned]
    if unassigned and batches:
        for idx in unassigned:
            pos = vn_in_away[np.where(vn_indices == idx)[0][0]]
            best_batch = min(batches.keys(),
                             key=lambda b: abs(pos - (bin_edges[b] + bin_edges[b + 1]) / 2))
            batches[best_batch].append(idx)

    return batches


def compute_batches_polar(DF, line1, line2, center_xy, center_idx, n_batches=30,
                          radius=None, tangent_deg=None):
    """Polar-based batching: divide by kernel-weighted distance to V1 boundary."""
    vn_mask = _get_vn_mask(DF)
    coords = DF[["x", "y"]].values.astype(float)
    vn_indices = np.where(vn_mask)[0]

    if len(vn_indices) == 0:
        return {}

    bnd_indices = np.concatenate([line1, line2])
    if len(bnd_indices) == 0:
        bnd_indices = np.where(DF["area"].values.astype(int) == 1)[0]

    vn_coords = coords[vn_indices]
    bnd_coords = coords[bnd_indices]

    vn_r = np.sqrt(np.sum((vn_coords - center_xy) ** 2, axis=1))
    vn_t = np.arctan2(vn_coords[:, 1] - center_xy[1],
                      vn_coords[:, 0] - center_xy[0])

    bnd_r = np.sqrt(np.sum((bnd_coords - center_xy) ** 2, axis=1))
    bnd_t = np.arctan2(bnd_coords[:, 1] - center_xy[1],
                       bnd_coords[:, 0] - center_xy[0])

    vn_r_2d = vn_r[:, None]
    bnd_r_2d = bnd_r[None, :]
    vn_t_2d = vn_t[:, None]
    bnd_t_2d = bnd_t[None, :]

    theta_diff = np.abs(vn_t_2d - bnd_t_2d)
    theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)

    r_min = np.minimum(vn_r_2d, bnd_r_2d)
    arc_length = r_min * theta_diff

    radius_length = np.abs(vn_r_2d - bnd_r_2d)

    if radius is not None and tangent_deg is not None:
        angle_rad = tangent_deg * np.pi / 180.0
        pair_dist = arc_length / angle_rad + radius_length / radius
    else:
        pair_dist = arc_length + radius_length

    polar_dist = np.min(pair_dist, axis=1)

    d_min, d_max = polar_dist.min(), polar_dist.max()
    bin_edges = np.linspace(d_min, d_max + 1e-10, n_batches + 1)

    batches = {}
    for b in range(n_batches):
        in_bin = (polar_dist >= bin_edges[b]) & (polar_dist < bin_edges[b + 1])
        batch_nodes = vn_indices[in_bin]
        if len(batch_nodes) > 0:
            batches[b] = batch_nodes.tolist()

    return batches


def compute_batches_euclidean(DF, line1, line2, center_xy, center_idx, n_batches=30):
    """
    Euclidean-distance-based batching: divide by distance from boundary lines.
    """
    vn_mask = _get_vn_mask(DF)
    coords = DF[["x", "y"]].values.astype(float)
    vn_indices = np.where(vn_mask)[0]

    if len(vn_indices) == 0:
        return {}

    vn_coords = coords[vn_indices]
    bnd_dist = _distance_to_boundary(vn_coords, DF, line1, line2)

    d_min, d_max = bnd_dist.min(), bnd_dist.max()
    bin_edges = np.linspace(d_min, d_max + 1e-10, n_batches + 1)

    batches = {}
    for b in range(n_batches):
        in_bin = (bnd_dist >= bin_edges[b]) & (bnd_dist < bin_edges[b + 1])
        batch_nodes = vn_indices[in_bin]
        if len(batch_nodes) > 0:
            batches[b] = batch_nodes.tolist()

    return batches


def compute_batches_x(DF, line1, line2, center_xy, center_idx, n_batches=30):
    """X-axis iso-line batching: rotate, divide into strips, rank by x-position."""
    vn_mask = _get_vn_mask(DF)
    coords = DF[["x", "y"]].values.astype(float)
    vn_indices = np.where(vn_mask)[0]

    if len(vn_indices) == 0:
        return {}

    mid_angle = _boundary_midangle(DF, line1, line2, center_xy)
    cos_a = np.cos(-mid_angle)
    sin_a = np.sin(-mid_angle)
    vn_coords = coords[vn_indices] - center_xy
    vn_rotated = np.column_stack([
        vn_coords[:, 0] * cos_a - vn_coords[:, 1] * sin_a,
        vn_coords[:, 0] * sin_a + vn_coords[:, 1] * cos_a
    ])

    x_vals = vn_rotated[:, 0]
    y_vals = vn_rotated[:, 1]

    y_step = 0.5
    y_min = np.floor(y_vals.min() / y_step) * y_step
    y_max = np.ceil(y_vals.max() / y_step) * y_step
    y_lines = np.arange(y_min, y_max + y_step * 0.5, y_step)
    n_strips = len(y_lines) - 1
    if n_strips <= 0:
        n_strips = 1
        y_lines = np.array([y_vals.min(), y_vals.max() + 1e-10])

    n_divisions = n_batches
    node_division_rank = np.full(len(vn_indices), -1, dtype=int)

    for s in range(n_strips):
        y_lo = y_lines[s]
        y_hi = y_lines[s + 1]
        if s < n_strips - 1:
            strip_mask = (y_vals >= y_lo) & (y_vals < y_hi)
        else:
            strip_mask = (y_vals >= y_lo) & (y_vals <= y_hi)

        strip_idx = np.where(strip_mask)[0]
        if len(strip_idx) == 0:
            continue

        strip_x = x_vals[strip_idx]
        x_min_s = strip_x.min()
        x_max_s = strip_x.max()
        x_range = x_max_s - x_min_s
        if x_range < 1e-10:
            node_division_rank[strip_idx] = 0
            continue

        x_edges = np.linspace(x_min_s, x_max_s + 1e-10, n_divisions + 1)
        for d in range(n_divisions):
            d_mask = (strip_x >= x_edges[d]) & (strip_x < x_edges[d + 1])
            node_division_rank[strip_idx[d_mask]] = d

    batches = {}
    for d in range(n_divisions):
        batch_nodes = vn_indices[node_division_rank == d]
        if len(batch_nodes) > 0:
            batches[d] = batch_nodes.tolist()

    unranked = vn_indices[node_division_rank < 0]
    if len(unranked) > 0 and batches:
        last_batch = max(batches.keys())
        batches[last_batch].extend(unranked.tolist())

    return batches


def order_within_batch(batch_node_indices, DF, center_xy, ecc_order="fp"):
    """Order nodes within a batch by eccentricity (fp=foveal first, pf=peripheral first)."""
    if len(batch_node_indices) == 0:
        return []

    coords = DF.iloc[batch_node_indices][["x", "y"]].values.astype(float)
    r = np.sqrt(np.sum((coords - center_xy) ** 2, axis=1))

    if ecc_order == "fp":
        order = np.argsort(r)
    elif ecc_order == "pf":
        order = np.argsort(-r)
    elif ecc_order == "random":
        order = np.random.permutation(len(batch_node_indices))
    else:
        raise ValueError(f"Unknown ecc_order: {ecc_order}. Choose from: fp, pf, random")

    return [batch_node_indices[i] for i in order]


_MODE_FUNCS = {
    "angle": compute_batches_angle,
    "polar": compute_batches_polar,
    "euclidean": compute_batches_euclidean,
    "x": compute_batches_x,
}


def parse_custom_mode(custom_batch_mode_str):
    """Parse e.g. 'polar_fp' into ('polar', 'fp')."""
    parts = custom_batch_mode_str.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid custom_batch_mode: '{custom_batch_mode_str}'. "
            f"Expected format: {{mode}}_{{fp|pf|random}} e.g. polar_fp"
        )
    mode = parts[0]
    ecc_order = parts[1]
    if mode not in _MODE_FUNCS:
        raise ValueError(f"Unknown batch mode '{mode}'. Choose from: {list(_MODE_FUNCS.keys())}")
    if ecc_order not in ("fp", "pf", "random"):
        raise ValueError(f"Unknown ecc_order '{ecc_order}'. Choose from: fp, pf, random")
    return mode, ecc_order


def get_custom_node_order(DF, mode, ecc_order, n_batches=30, radius=None, tangent_deg=None):
    """Compute custom batch assignments and node visitation order."""
    line1, line2, center_xy, center_idx = compute_v1_boundary(DF)
    print(f"[CustomBatch] V1 boundary: line1={len(line1)} nodes, line2={len(line2)} nodes")
    print(f"[CustomBatch] Center: idx={center_idx}, xy={center_xy}")

    batch_func = _MODE_FUNCS[mode]
    if mode == "polar" and radius is not None and tangent_deg is not None:
        batches = batch_func(DF, line1, line2, center_xy, center_idx, n_batches,
                             radius=radius, tangent_deg=tangent_deg)
    else:
        batches = batch_func(DF, line1, line2, center_xy, center_idx, n_batches)
    print(f"[CustomBatch] Mode={mode}: {len(batches)} non-empty batches")

    ordered_batches = {}
    total_ordered = []
    for batch_id in sorted(batches.keys()):
        ordered_nodes = order_within_batch(batches[batch_id], DF, center_xy, ecc_order)
        ordered_batches[batch_id] = ordered_nodes
        total_ordered.extend(ordered_nodes)

    print(f"[CustomBatch] Total ordered Vn nodes: {len(total_ordered)}")

    vn_mask = DF["area"].values.astype(int) != 1
    expected_count = int(np.sum(vn_mask))
    actual_count = len(set(total_ordered))
    if actual_count != expected_count:
        print(f"[CustomBatch] WARNING: expected {expected_count} Vn nodes, got {actual_count}")

    return total_ordered, ordered_batches
