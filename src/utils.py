import os
import torch
import numpy as np
import pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

sys.path.append('..')
from node_color_utils import calculate_node_colors
# Import unified tuning color utilities
import sys
sys.path.insert(0, '..')  # Add parent directory to path
from TUNING_COLOR_UTILS import compute_tuning_colors

# Color threshold for masking V1 nodes: nodes with color <= COLOR_MASK_THRESHOLD are masked
# This must match COLOR_MASK_THRESHOLD in polarModel.py
# COLOR_MASK_THRESHOLD = 0.2  # DISABLED: color mask strategy not used


def normalize_angle(a):
    """Normalize angle to [0, 360)."""
    return a % 360

def get_v2_v4_range(deg):
    """
    Find the angular range that covers all V2-V4 nodes by identifying the largest gap.
    The largest gap corresponds to the V1 region (which we want to exclude/sweep away from).
    
    Returns: (start_angle, end_angle, sweep_width)
    - start_angle: The clockwise start of the V2-V4 block (end of the V1 gap).
    - end_angle: The clockwise end of the V2-V4 block (start of the V1 gap).
    - sweep_width: The angular width of the V2-V4 block (360 - max_gap).
    """
    if len(deg) == 0:
        return 0, 0, 0
        
    deg_sorted = np.sort(deg)
    # Differences between consecutive angles
    diffs = np.diff(deg_sorted)
    # Wrap-around difference
    wrap_diff = 360 - (deg_sorted[-1] - deg_sorted[0])
    
    all_diffs = np.append(diffs, wrap_diff)
    max_gap_idx = np.argmax(all_diffs)
    max_gap = all_diffs[max_gap_idx]
    
    # The gap is between deg_sorted[max_gap_idx] and deg_sorted[max_gap_idx+1]
    # (handling wrap around index)
    
    if max_gap_idx < len(diffs):
        gap_start = deg_sorted[max_gap_idx]
        gap_end = deg_sorted[max_gap_idx + 1]
    else:
        # Wrap around gap: last element to first element
        gap_start = deg_sorted[-1]
        gap_end = deg_sorted[0]
        
    # The V2-V4 block is the COMPLEMENT of the gap.
    # It starts at gap_end and ends at gap_start.
    
    start_angle = normalize_angle(gap_end)
    end_angle = normalize_angle(gap_start)
    sweep_width = 360 - max_gap
    
    return start_angle, end_angle, sweep_width

def count_in_interval_mask(deg, start, end):
    s = normalize_angle(start)
    e = normalize_angle(end)
    if s <= e:
        return (deg > s) & (deg <= e)
    else:
        return (deg > s) | (deg <= e)

def compute_dynamic_batch_sizes(DF, output_dir=None, data_name="data", tag_name="tag"):
    """
    Compute batch size schedule dynamically ensuring 100% coverage of V2-V4 nodes.
    Strategy:
    1. Identify V2-V4 angular block by finding the largest gap (V1).
    2. Sweep outward from the edges of V1 into the V2-V4 block.
    3. Count V2-V4 nodes in 2-degree non-overlapping steps.
    
    Returns: list of batch sizes (integers).
    Optionally saves to text file if output_dir is provided.
    """
    # 1. Get V2-V4 angles
    vn = DF[DF["area"] != 1]
    # DF['t'] stores angular position in radians.
    vn_t = vn["t"].values
    vn_deg = normalize_angle(np.degrees(vn_t))
    total_vn = len(vn_deg)
    
    if total_vn == 0:
        return [1]

    # 2. Find range
    start_angle, end_angle, sweep_width = get_v2_v4_range(vn_deg)
    
    # 3. Sweep
    # Sweep inward across the V2-V4 block from its two angular endpoints.
    
    step_size = 2.0
    front_plus = start_angle
    front_minus = end_angle
    
    counts = []
    
    # We sweep until the fronts meet.
    # The total distance to cover is sweep_width.
    # Each step covers 2*step_size (one from each side).
    
    max_steps = int(np.ceil(sweep_width / (2 * step_size))) + 2
    
    covered_mask = np.zeros(total_vn, dtype=bool)
    
    for i in range(max_steps):
        next_plus = normalize_angle(front_plus + step_size)
        next_minus = normalize_angle(front_minus - step_size)
        
        # Count in (front_plus, next_plus] and (next_minus, front_minus]
        in1 = count_in_interval_mask(vn_deg, front_plus, next_plus)
        in2 = count_in_interval_mask(vn_deg, next_minus, front_minus)
        
        # Union of new nodes
        new_mask = (in1 | in2) & (~covered_mask)
        count = int(np.sum(new_mask))
        counts.append(count)
        
        covered_mask = covered_mask | new_mask
        
        front_plus = next_plus
        front_minus = next_minus
        
        if np.all(covered_mask):
            break

    # If explicit file save requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"batch_size_{data_name}_{tag_name}.txt")
        with open(fname, "w") as f:
            for c in counts:
                f.write(f"{c}\n")
        print(f"Computed dynamic batch sizes (Total Vn={total_vn}, Sum={sum(counts)}). Saved to {fname}")
        
    return counts

def initDirectory(param, outputDir):
    """Lightweight replacement for the legacy initDirectory helper.

    The current pipeline saves all results from experiment-level code,
    so the model only needs a placeholder output directory.
    """
    # Keep behavior minimal to avoid side effects; caller can ignore.
    return outputDir


def computeV2V4MSE(DF, W):
    """Compute MSE between predicted and true V2–V4 tuning.

    This is the canonical implementation used by both the model
    (`polarModel.VisualMatrix3D`) and experiment scripts.
    """
    V1_count = len(DF[DF["area"] == 1])
    Vn_W = W[:, V1_count:]  # V1 to Vn connections

    # Get V1 tuning vectors
    V1_df = DF[DF["area"] == 1].copy()
    V1_tuning = V1_df[["tuningX", "tuningY"]].values

    # Convert tensors to numpy if needed
    if hasattr(Vn_W, "cpu"):  # Check if it's a PyTorch tensor
        Vn_W = Vn_W.cpu().numpy()
    if hasattr(V1_tuning, "cpu"):  # Check if it's a PyTorch tensor
        V1_tuning = V1_tuning.cpu().numpy()

    # Predict V2–V4 tuning
    predicted_tuning = Vn_W.T @ V1_tuning  # (Vn_count, 2)

    # Get true V2–V4 tuning
    Vn_df = DF[DF["area"] != 1].copy()
    true_tuning = Vn_df[["tuningX", "tuningY"]].values

    # Compute MSE
    mse = np.mean((predicted_tuning - true_tuning) ** 2)

    # Return MSE-only metrics
    return mse, mse, None


def _rotate_to_align_x(xs: np.ndarray, ys: np.ndarray, areas: np.ndarray):
    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    areas_np = np.asarray(areas, dtype=int)
    unique_areas = np.unique(areas_np)
    centroids = []
    for a in unique_areas:
        mask = areas_np == a
        if not np.any(mask):
            continue
        centroids.append([xs_np[mask].mean(), ys_np[mask].mean()])
    if len(centroids) < 2:
        return xs_np, ys_np, 0.0, np.array([xs_np.mean() if xs_np.size else 0.0, ys_np.mean() if ys_np.size else 0.0])
    C = np.array(centroids, dtype=float)
    C_centered = C - C.mean(axis=0)
    cov = np.cov(C_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.arctan2(principal[1], principal[0])
    cos_t, sin_t = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    P = np.stack([xs_np, ys_np], axis=1) @ R.T
    center_rot = (C.mean(axis=0)) @ R.T
    return P[:, 0], P[:, 1], angle, center_rot

def _rotate_by_angle(xs: np.ndarray, ys: np.ndarray, delta_rad: float):
    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    R = np.array([[c, -s], [s, c]])
    P = np.stack([xs_np, ys_np], axis=1) @ R.T
    return P[:, 0], P[:, 1]

def plot_tuning_compare_two_panel(
    DF,
    true_colors_array,
    pred_colors_array,
    args,
    param_suffix="",
    masked_v1_indices=None,
    unconnected_vn_indices=None,
    pred_tuning_coords=None,
    distance_mode="polar",
    ref_ecc_colors=None,
):
    """
    Create and save a two-panel tuning comparison plot:
    - Left: true tuning colors
    - Right: predicted tuning colors (V1 uses true colors)
    - Masked V1 nodes (color < 0.02) are highlighted with gray edges
    - Unconnected V2-V4 nodes (no connections) are shown in black
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    coords = DF[["x", "y"]].values  # already aligned if mode==mds
    areas = DF["area"].values.astype(int)

    # Build color arrays aligned to node order using discrete colormap (0.1 unit steps)
    # Use colormap + flip heuristic from TUNING_COLOR_UTILS (canonical)
    from TUNING_COLOR_UTILS import get_tuning_colormap, should_flip_y_red_bottom
    cmap = get_tuning_colormap()
    # Round colors to nearest 0.1 for discrete colormap
    true_colors_discrete = np.round(np.array(true_colors_array) * 10) / 10.0
    true_colors_discrete = np.clip(true_colors_discrete, 0.0, 1.0)
    pred_colors_discrete = np.round(np.array(pred_colors_array) * 10) / 10.0
    pred_colors_discrete = np.clip(pred_colors_discrete, 0.0, 1.0)
    true_rgba = [cmap(c) for c in true_colors_discrete]
    pred_rgba = [cmap(c) for c in pred_colors_discrete]

    # Enforce orientation: red should be bottom, blue top.
    # If not satisfied, flip across x-axis (y -> -y). This matches the hybrid comparison pipeline.
    # For rotated datasets (e.g., R1_gpr_grid_90_lh), use fixed flip based on tag:
    #   lh: no flip (flip_y = False)
    #   rh: yes flip (flip_y = True)
    v1_mask = (areas == 1)
    
    # Check if this is a rotated dataset (contains _90, _180, or _270)
    import re
    data_name = args.data
    is_rotated = bool(re.search(r'_(90|180|270)(_|$)', data_name))  # Matches _90, _180, or _270 followed by _ or end
    
    if is_rotated:
        # For rotated datasets, use fixed flip based on tag
        if args.tag == "lh":
            flip_y = False  # lh: no flip
        elif args.tag == "rh":
            flip_y = True   # rh: yes flip
        else:
            # Fallback to normal flip logic
            try:
                flip_y = should_flip_y_red_bottom(
                    coords[v1_mask] if np.any(v1_mask) else coords,
                    np.asarray(true_colors_discrete, dtype=float)[v1_mask] if np.any(v1_mask) else np.asarray(true_colors_discrete, dtype=float),
                )
            except Exception:
                flip_y = False
    else:
        # For non-rotated datasets, use normal flip logic
        try:
            flip_y = should_flip_y_red_bottom(
                coords[v1_mask] if np.any(v1_mask) else coords,
                np.asarray(true_colors_discrete, dtype=float)[v1_mask] if np.any(v1_mask) else np.asarray(true_colors_discrete, dtype=float),
            )
        except Exception:
            flip_y = False
    
    if flip_y:
        coords = coords.copy()
        coords[:, 1] *= -1.0

    # Convert masked_v1_indices to set for fast lookup
    if masked_v1_indices is None:
        masked_v1_set = set()
    else:
        masked_v1_set = set(masked_v1_indices)
    
    # Convert unconnected_vn_indices to set for fast lookup
    if unconnected_vn_indices is None:
        unconnected_vn_set = set()
    else:
        unconnected_vn_set = set(unconnected_vn_indices)
    
    # Find is_center node (should be at origin after alignment)
    is_center_idx = None
    if "is_center" in DF.columns:
        center_mask = DF["is_center"].values.astype(int) == 1
        if np.any(center_mask):
            is_center_idx = int(np.where(center_mask)[0][0])

    # Output directory for tuning-comparison plots
    out_base = "../outputs/plots"
    os.makedirs(out_base, exist_ok=True)
    out_path = os.path.join(out_base, f"{args.data}_{args.tag}_tuning_compare{param_suffix}.png")

    # Compact three-panel plot (true / predicted / eccentricity).
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    # Use a common marker style across areas.
    marker_sym = 'o'
    marker_size = 10

    # Helper function to plot with masked V1 highlighting
    def plot_with_mask(ax, coords, areas, colors, title, is_predicted_panel=False):
        """
        is_predicted_panel: If True, show unconnected V2-V4 nodes in black (for predicted panel).
                           If False (for true tuning panel), show all nodes with their colors.
        """
        for a in np.unique(areas):
            idxs = np.where(areas == a)[0]
            if idxs.size == 0:
                continue
            
            # Separate masked and unmasked nodes for area 1
            if int(a) == 1:
                unmasked_idxs = [i for i in idxs if i not in masked_v1_set]
                masked_idxs = [i for i in idxs if i in masked_v1_set]
                
                # Plot unmasked V1 nodes normally
                if unmasked_idxs:
                    ax.scatter(
                        coords[unmasked_idxs, 0], coords[unmasked_idxs, 1],
                        c=[colors[i] for i in unmasked_idxs],
                        s=marker_size, alpha=1.0, linewidth=0,
                        marker=marker_sym,
                        label=f"Area {int(a)}"
                    )
                
                # Plot masked V1 nodes with gray edge
                if masked_idxs:
                    ax.scatter(
                        coords[masked_idxs, 0], coords[masked_idxs, 1],
                        c=[colors[i] for i in masked_idxs],
                        s=marker_size, alpha=1.0, linewidth=0.5, edgecolors='gray',
                        marker=marker_sym,
                    )
            else:
                # Non-V1 areas: check for unconnected nodes
                connected_idxs = [i for i in idxs if i not in unconnected_vn_set]
                unconnected_idxs = [i for i in idxs if i in unconnected_vn_set]
                
                # Plot connected nodes normally
                if connected_idxs:
                    ax.scatter(
                        coords[connected_idxs, 0], coords[connected_idxs, 1],
                        c=[colors[i] for i in connected_idxs],
                        s=marker_size, alpha=1.0, linewidth=0,
                        marker=marker_sym,
                        label=f"Area {int(a)}"
                    )
                
                # Plot unconnected nodes
                if unconnected_idxs:
                    if is_predicted_panel:
                        # Predicted panel: show unconnected nodes in black
                        ax.scatter(
                            coords[unconnected_idxs, 0], coords[unconnected_idxs, 1],
                            c='black',
                            s=marker_size, alpha=1.0, linewidth=0,
                            marker=marker_sym,
                        )
                    else:
                        # True panel: show unconnected nodes with their true colors
                        ax.scatter(
                            coords[unconnected_idxs, 0], coords[unconnected_idxs, 1],
                            c=[colors[i] for i in unconnected_idxs],
                            s=marker_size, alpha=1.0, linewidth=0,
                            marker=marker_sym,
                        )
        ax.set_title(title)
        ax.set_aspect('equal')
        # Remove background grid and axes.
        ax.axis('off')
        
        # Plot is_center node with black bold X marker
        if is_center_idx is not None:
            ax.scatter(
                coords[is_center_idx, 0], coords[is_center_idx, 1],
                c='black',
                s=100, alpha=1.0, linewidth=2,
                marker='x',
                zorder=10  # Ensure it's on top
            )

    # Left: true (always show all nodes with true colors)
    plot_with_mask(axes[0], coords, areas, true_rgba, "True", is_predicted_panel=False)

    # Right: predicted (show unconnected nodes in black).
    plot_with_mask(axes[1], coords, areas, pred_rgba, "Polar angle", is_predicted_panel=True)

    # Third: eccentricity (r-based) colors.
    try:
        if ref_ecc_colors is not None:
            # Use pre-computed eccentricity from full-dataset reference
            pred_r = np.asarray(ref_ecc_colors, dtype=float)
        else:
            # Standard: compute from tuning vectors using V1 quantiles
            from TUNING_COLOR_UTILS import compute_tuning_colors_r, round_color_bins
            true_tuning_coords = DF[["tuningX", "tuningY"]].values.astype(float)
            if pred_tuning_coords is None:
                pred_tuning_coords_eff = true_tuning_coords
            else:
                pred_tuning_coords_eff = np.asarray(pred_tuning_coords, dtype=float)

            true_r = round_color_bins(np.asarray(compute_tuning_colors_r(true_tuning_coords, v1_mask=v1_mask, tag=args.tag), dtype=float))
            pred_r = round_color_bins(np.asarray(compute_tuning_colors_r(pred_tuning_coords_eff, v1_mask=v1_mask, tag=args.tag), dtype=float))
            pred_r[v1_mask] = true_r[v1_mask]
        pred_r = np.clip(pred_r, 0.0, 0.9)

        ecc_rgba = [cmap(float(c)) for c in pred_r]
        plot_with_mask(axes[2], coords, areas, ecc_rgba, "Eccentricity", is_predicted_panel=False)
    except Exception as e:
        axes[2].set_title("Eccentricity")
        axes[2].axis("off")
        print(f"Warning: eccentricity subplot failed: {e}")

    # Remove legends for a clean look
    for ax in axes:
        if ax.get_legend():
            ax.legend().remove()

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Three-panel plot saved to {out_path}")


def loadDataDF(data="X1", tag="lh", mode="sphere"):
    """Load fMRI PKL and build the aligned DataFrame used by the model.

    This is the single canonical implementation; other modules should import
    this from `utils` instead of defining their own copies.
    """
    print(f"Loading fMRI data from pkl file: {data}_{tag} in {mode} mode...")

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device available: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Construct pkl file path based on data and tag.
    # The data root can be overridden through SHARED_DATA_ROOT.
    _repo_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    _data_root = os.environ.get("SHARED_DATA_ROOT", _repo_data)
    pkl_file = f"{_data_root}/{data}_{tag}.pkl"
    if not os.path.exists(pkl_file):
        pkl_file = f"../data/{data}_{tag}.pkl"

    # Load pkl file from parent directory
    with open(pkl_file, "rb") as file:
        fMRI_data = pickle.load(file)

    print(f"Loaded {len(fMRI_data)} nodes from pkl file ({data}_{tag})")

    # Extract data based on mode
    area = []
    x, y, z = [], [], []  # Generic coordinates based on mode
    x_mds, y_mds = [], []  # MDS coordinates for visualization
    sx, sy, sz = [], [], []  # raw (uncentered) sphere coords for sphere_geo distance
    tx, ty = [], []
    is_center_flags = []
    ID = []
    nodeIdx = []

    for item in fMRI_data:
        value = fMRI_data[item]
        nodeIdx.append(int(item))
        area.append(int(float(value["area"])))

        sph_loc = value.get("loc_sphere", None)
        if sph_loc is not None:
            sx.append(float(sph_loc[0]))
            sy.append(float(sph_loc[1]))
            sz.append(float(sph_loc[2]))
        else:
            sx.append(0.0)
            sy.append(0.0)
            sz.append(0.0)

        # Load mode-specific coordinates for distance calculation
        if mode == "sphere":
            loc = value["loc_sphere"]
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])
        elif mode == "euclidean":
            loc = value["loc_3D"]
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])
        elif mode == "mds":
            loc = value["loc"]
            x.append(loc[0])
            y.append(loc[1])
            z.append(0.0)  # Default z value for 2D data
        else:
            # Default to sphere mode
            loc = value["loc_sphere"]
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])

        # Always load MDS coordinates for visualization
        mds_loc = value["loc"]
        x_mds.append(mds_loc[0])
        y_mds.append(mds_loc[1])

        tx.append(float(value["tuning"][0]))
        ty.append(float(value["tuning"][1]))
        # Use nodeIdx as ID directly since nodeOrder is not needed
        ID.append(int(item))
        # Optional center flag (present in interpolated PKLs)
        is_center_flags.append(int(value.get("is_center", 0)))

    # Create DataFrame with both mode-specific and MDS coordinates
    DF = pd.DataFrame(
        {
            "nodeIdx": nodeIdx,
            "ID": ID,
            "area": area,
            "x": x,
            "y": y,
            "z": z,  # Mode-specific coordinates for distance calculation
            "x_mds": x_mds,
            "y_mds": y_mds,  # MDS coordinates for visualization
            "sx": sx,
            "sy": sy,
            "sz": sz,  # raw sphere coords (radius ~100) for sphere_geo kernel
            "tuningX": tx,
            "tuningY": ty,
            "is_center": is_center_flags,
        }
    ).astype({"ID": int, "area": int})

    print(f"Created DataFrame with shape: {DF.shape}")
    print(f"Area distribution: {DF['area'].value_counts().sort_index().to_dict()}")
    print(f"Coordinate mode: {mode}")

    # Similar calculations to dataProcess.py transformData
    # Separate V1 and non-V1 areas
    V1DF = DF[DF["area"] == 1].copy()
    VnDF = DF[DF["area"] != 1].copy()
    V1Count = V1DF.shape[0]
    VnCount = VnDF.shape[0]

    # Skip distance calculation for now - set default values
    VnDF.loc[:, "V1Dist"] = 0.0
    VnDF.loc[:, "distGroup"] = pd.cut([0.0] * VnCount, 10)
    VnDF.loc[:, "anchorOrder"] = list(range(VnCount))

    V1DF.loc[:, "V1Dist"] = -1
    V1DF.loc[:, "distGroup"] = "(0)"
    V1DF.loc[:, "anchorOrder"] = -1

    # Combine data
    temp = pd.concat([V1DF, VnDF])

    # Tuning-related calculations
    temp.loc[:, "tuningX"] = temp["tuningX"].astype(float)
    temp.loc[:, "tuningY"] = temp["tuningY"].astype(float)
    temp.loc[:, "tuningR"] = np.sqrt(np.square(temp["tuningX"]) + np.square(temp["tuningY"]))
    temp.loc[:, "tuningT"] = np.arctan2(temp["tuningY"], temp["tuningX"])

    # Position-related calculations with alignment/translation
    # Use is_center node to align coordinates for all modes
    center_mask = temp["is_center"].values.astype(int) == 1
    if not np.any(center_mask):
        raise RuntimeError("No is_center==1 node found in PKL; cannot set center.")
    center_idx = int(np.where(center_mask)[0][0])
    
    if mode == "mds":
        # Apply PCA-based alignment for MDS mode
        xs = temp["x"].values.astype(float)
        ys = temp["y"].values.astype(float)
        areas_arr = temp["area"].values.astype(int)
        xs_rot, ys_rot, _, _ = _rotate_to_align_x(xs, ys, areas_arr)

        # Rotate so V1 centroid points left
        mask_a1 = areas_arr == 1
        if np.any(mask_a1):
            cx = xs_rot[mask_a1].mean()
            cy = ys_rot[mask_a1].mean()
            cur_angle = np.arctan2(cy, cx)
            delta = np.pi - cur_angle
            xs_rot, ys_rot = _rotate_by_angle(xs_rot, ys_rot, delta)

        # Use the node marked is_center==1 as the center (should be exactly one)
        c_x = float(xs_rot[center_idx])
        c_y = float(ys_rot[center_idx])

        xs_aligned = xs_rot - c_x
        ys_aligned = ys_rot - c_y

        temp.loc[:, "x"] = xs_aligned
        temp.loc[:, "y"] = ys_aligned
        temp.loc[:, "center_x"] = c_x
        temp.loc[:, "center_y"] = c_y
        print(f"Center (from is_center) for {data}_{tag} in {mode} mode: ({c_x:.4f}, {c_y:.4f})")
    else:
        # For other modes (sphere, euclidean), align using is_center node
        xs = temp["x"].values.astype(float)
        ys = temp["y"].values.astype(float)
        zs = temp["z"].values.astype(float)
        
        c_x = float(xs[center_idx])
        c_y = float(ys[center_idx])
        c_z = float(zs[center_idx])
        
        xs_aligned = xs - c_x
        ys_aligned = ys - c_y
        zs_aligned = zs - c_z
        
        temp.loc[:, "x"] = xs_aligned
        temp.loc[:, "y"] = ys_aligned
        temp.loc[:, "z"] = zs_aligned
        temp.loc[:, "center_x"] = c_x
        temp.loc[:, "center_y"] = c_y
        print(f"Center (from is_center) for {data}_{tag} in {mode} mode: ({c_x:.4f}, {c_y:.4f}, {c_z:.4f})")

    # Compute r, t from translated x,y (for all modes; sphere uses it for distance)
    temp.loc[:, "r"] = np.sqrt(np.square(temp["x"]) + np.square(temp["y"]))
    temp.loc[:, "t"] = np.arctan2(temp["y"], temp["x"])

    temp.loc[:, "tuningTAlt"] = np.absolute(temp["t"]) * -1

    # Initialize boundary
    temp.loc[:, "Boundary"] = 0
    temp.loc[:, "Weight"] = 0

    # Set boundary based on tuning
    tuningMin = np.min(np.absolute(temp["tuningT"]))
    temp.loc[((temp["area"] == 1) & (np.absolute(temp["tuningT"] - tuningMin) <= 5e-2)), "Boundary"] = 1
    temp.loc[temp["Boundary"] > 0, "Weight"] = 1

    # Sort and reassign ID
    temp = temp.sort_values(by=["area", "ID"]).reset_index(drop=True)
    temp.loc[:, "ID"] = temp.index

    return temp


def save_baseline_results(
    DF,
    W,
    args,
    actual_params,
    spatial_results=None,
    param_suffix="",
    node_generation_order=None,
    batch_info=None,
    tsv_only: bool = False,
    distance_mode: str = "polar",
    ref_colors_path: str = None,
):
    """Save prediction results (tuning TSV, weight NPZ) and the comparison plot.
    Always generates the plot and returns pred_colors_array for video generation.
    """
    import os
    import numpy as np
    import pandas as pd

    # Save to a single fixed base directory
    output_base_dir = "../outputs/predictions"
    if distance_mode == "euclidean":
        mode_dir = os.path.join(output_base_dir, "euclidean")
    else:
        mode_dir = os.path.join(output_base_dir, args.mode)
    os.makedirs(mode_dir, exist_ok=True)

    # File naming convention: {type}_{data}_{tag}_{algo}_{repeat_idx}.{ext}
    base_filename = f"{args.data}_{args.tag}_{args.algo}{param_suffix}"

    # Convert tensors to numpy if needed for all operations
    if hasattr(W, "cpu"):  # Check if weight matrix is a PyTorch tensor
        W_numpy = W.cpu().numpy()
    else:
        W_numpy = W

    # 1. Save predicted tuning (only V2–V4 nodes)
    V1_count = len(DF[DF["area"] == 1])
    Vn_W = W[:, V1_count:]  # V1 to Vn connections

    # Get V1 tuning vectors for prediction
    V1_df = DF[DF["area"] == 1].copy()
    V1_tuning_vectors = V1_df[["tuningX", "tuningY"]].values

    # Convert tensors to numpy if needed
    if hasattr(Vn_W, "cpu"):
        Vn_W = Vn_W.cpu().numpy()
    if hasattr(V1_tuning_vectors, "cpu"):
        V1_tuning_vectors = V1_tuning_vectors.cpu().numpy()

    # Predict V2–V4 tuning using column-normalized weight matrix (weighted average of connected V1 tunings)
    col_sums = np.sum(Vn_W, axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    Vn_W_norm = Vn_W / col_sums
    predicted_tuning = Vn_W_norm.T @ V1_tuning_vectors  # (Vn_count, 2)

    # Get true tuning for V2–V4
    Vn_df = DF[DF["area"] != 1].copy()
    true_tuning = Vn_df[["tuningX", "tuningY"]].values

    tuning_data = []
    for idx, (_, node) in enumerate(Vn_df.iterrows()):
        tuning_data.append(
            {
                "Node_ID": int(node["nodeIdx"]),
                "Pred_0": float(predicted_tuning[idx, 0]),
                "Pred_1": float(predicted_tuning[idx, 1]),
                "True_0": float(true_tuning[idx, 0]),
                "True_1": float(true_tuning[idx, 1]),
            }
        )

    tuning_df = pd.DataFrame(tuning_data)
    tuning_file = os.path.join(mode_dir, f"predicted_{base_filename}.tsv")
    tuning_df.to_csv(tuning_file, sep="\t", index=False)

    # If requested, stop after writing the TSV.
    if tsv_only:
        print("Results saved (TSV-only):")
        print(f"   Tuning: {tuning_file}")
        return actual_params["mse"], None

    # 2. Save weight matrix (with node_generation_order if provided)
    weight_file = os.path.join(mode_dir, f"W_{base_filename}.npz")
    save_dict = {"W": W_numpy}
    if node_generation_order is not None:
        node_order_np = np.array(node_generation_order, dtype=np.int32)
        save_dict["node_generation_order"] = node_order_np
    if batch_info is not None:
        # batch_info is a list of batches, each batch is a list of (vn_idx, [v1_indices], predicted_tuning_color)
        # Save using pickle
        import pickle
        batch_info_bytes = pickle.dumps(batch_info)
        save_dict["batch_info"] = np.array([batch_info_bytes], dtype=object)
    np.savez_compressed(weight_file, **save_dict)

    # 3. Params CSV (disabled - not used)
    param_file = None

    # 4. Two-panel plot: left=true, right=predicted (area1 uses true) - ALWAYS GENERATED
    pred_colors_array = None
    try:
        import matplotlib.pyplot as plt
        node_ids = DF["nodeIdx"].values.astype(int)

        # Build full true/pred arrays aligned to node order
        true_map = {
            int(row["nodeIdx"]): np.array(
                [float(row["tuningX"]), float(row["tuningY"])], dtype=float
            )
            for _, row in DF.iterrows()
        }
        pred_map = {}
        # Fill area==1 with true tuning
        for _, row in DF[DF["area"] == 1].iterrows():
            nid = int(row["nodeIdx"])
            pred_map[nid] = np.array(
                [float(row["tuningX"]), float(row["tuningY"])], dtype=float
            )
        # Fill area!=1 from predicted_tuning aligned with Vn_df
        for idx, (_, row) in enumerate(Vn_df.iterrows()):
            nid = int(row["nodeIdx"])
            pred_map[nid] = np.array(
                [float(predicted_tuning[idx, 0]), float(predicted_tuning[idx, 1])],
                dtype=float,
            )

        # Build full tuning coordinate arrays for all nodes
        true_tuning_coords = np.array(
            [true_map.get(int(nid), [0.0, 0.0]) for nid in node_ids], dtype=float
        )
        pred_tuning_coords = np.array(
            [pred_map.get(int(nid), [0.0, 0.0]) for nid in node_ids], dtype=float
        )
        
        # V1 mask: indicates which nodes are V1
        v1_mask = DF["area"].values == 1
        v1_indices = np.where(v1_mask)[0]

        if ref_colors_path is not None:
            # --- 5-bin split mode: use pre-computed colors from full dataset ---
            ref = np.load(ref_colors_path, allow_pickle=True)
            ref_map = dict(zip(ref["nodeIdx"].astype(int), ref["colors"].astype(float)))
            available_bins = ref["available_bins"].astype(float)

            # Override DF x,y with full-dataset aligned coordinates
            if "x_aligned" in ref and "y_aligned" in ref:
                ref_x_map = dict(zip(ref["nodeIdx"].astype(int), ref["x_aligned"].astype(float)))
                ref_y_map = dict(zip(ref["nodeIdx"].astype(int), ref["y_aligned"].astype(float)))
                for i, nid in enumerate(node_ids):
                    nid_int = int(nid)
                    if nid_int in ref_x_map:
                        DF.iloc[i, DF.columns.get_loc("x")] = ref_x_map[nid_int]
                        DF.iloc[i, DF.columns.get_loc("y")] = ref_y_map[nid_int]
                print("Overrode DF x,y with full-dataset aligned coordinates")

            # Eccentricity: load from reference
            ref_ecc_map = None
            if "eccentricity" in ref:
                ref_ecc_map = dict(zip(ref["nodeIdx"].astype(int), ref["eccentricity"].astype(float)))

            # True colors: look up each node from reference
            true_colors_array = np.array(
                [ref_map.get(int(nid), available_bins[len(available_bins)//2]) for nid in node_ids],
                dtype=float,
            )

            # Predicted V2-V4 colors: weighted average of connected V1 colors through W
            V1_colors_scalar = true_colors_array[v1_indices]  # (V1_count,)
            V1_count_ref = len(v1_indices)
            Vn_W_ref = W_numpy[:, V1_count_ref:]
            col_sums_ref = np.sum(Vn_W_ref, axis=0, keepdims=True)
            col_sums_ref[col_sums_ref == 0] = 1.0
            Vn_W_norm_ref = Vn_W_ref / col_sums_ref
            pred_vn_colors = Vn_W_norm_ref.T @ V1_colors_scalar  # (Vn_count,)
            pred_vn_colors = np.round(pred_vn_colors * 10.0) / 10.0
            pred_vn_colors = np.clip(pred_vn_colors, available_bins.min(), available_bins.max())

            # Build full pred array: V1 = true, V2-V4 = inherited
            pred_colors_array = true_colors_array.copy()
            vn_indices = np.where(~v1_mask)[0]
            for i, df_idx in enumerate(vn_indices):
                pred_colors_array[df_idx] = pred_vn_colors[i]

            # Build eccentricity: V1 = ref, V2-V4 = inherited through W
            ref_ecc_colors = None
            if ref_ecc_map is not None:
                true_ecc = np.array(
                    [ref_ecc_map.get(int(nid), 0.5) for nid in node_ids], dtype=float,
                )
                V1_ecc_scalar = true_ecc[v1_indices]
                pred_vn_ecc = Vn_W_norm_ref.T @ V1_ecc_scalar
                pred_vn_ecc = np.round(pred_vn_ecc * 10.0) / 10.0
                pred_vn_ecc = np.clip(pred_vn_ecc, 0.0, 0.9)
                ref_ecc_colors = true_ecc.copy()
                ref_ecc_colors[v1_indices] = true_ecc[v1_indices]
                for i, df_idx in enumerate(vn_indices):
                    ref_ecc_colors[df_idx] = pred_vn_ecc[i]

            print(f"Using ref_colors ({len(available_bins)} bins: {available_bins})")
        else:
            # --- Standard 10-bin mode ---
            # Compute colors with V1-true-defined bins (canonical):
            # compute_tuning_colors() defines bin boundaries using V1 only, then applies to all nodes.
            true_colors_array = compute_tuning_colors(true_tuning_coords, v1_mask=v1_mask, tag=args.tag)
            pred_colors_array = compute_tuning_colors(pred_tuning_coords, v1_mask=v1_mask, tag=args.tag)

            # V1 nodes use true colors for both true and pred
            pred_colors_array[v1_indices] = true_colors_array[v1_indices]

        # DISABLED: color mask strategy not used
        # # Find masked V1 nodes (color <= COLOR_MASK_THRESHOLD) for gray edge highlighting
        # masked_v1_indices = []
        # for idx in v1_indices:
        #     if true_colors_array[idx] <= COLOR_MASK_THRESHOLD:
        #         masked_v1_indices.append(idx)
        masked_v1_indices = []  # No masked nodes

        # Find V2-V4 nodes with no connections (col_sum == 0)
        V1_count = len(v1_indices)
        Vn_W = W_numpy[:, V1_count:]  # V1->Vn connections
        col_sums = np.sum(Vn_W, axis=0)
        unconnected_vn_indices = []
        for idx, (_, row) in enumerate(Vn_df.iterrows()):
            vn_col_idx = idx  # Vn_df is already filtered to area != 1
            if col_sums[vn_col_idx] == 0:
                # Find the index in full DF
                df_idx = DF[DF["nodeIdx"] == row["nodeIdx"]].index[0]
                unconnected_vn_indices.append(df_idx)

        plot_tuning_compare_two_panel(
            DF, true_colors_array, pred_colors_array, args,
            param_suffix=param_suffix,
            masked_v1_indices=masked_v1_indices if masked_v1_indices else None,
            unconnected_vn_indices=unconnected_vn_indices if unconnected_vn_indices else None,
            pred_tuning_coords=pred_tuning_coords,
            distance_mode=distance_mode,
            ref_ecc_colors=ref_ecc_colors if ref_colors_path is not None else None,
        )
    except Exception as e:
        print(f"Warning: two-panel plot failed: {e}")

    print("Results saved:")
    print(f"   Tuning: {tuning_file}")
    print(f"   Weights: {weight_file}")
    if param_file:
        print(f"   Params: {param_file}")

    return actual_params["mse"], pred_colors_array




