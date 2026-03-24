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
from node_color_utils import calculate_node_colors_newcode_style
from TUNING_COLOR_UTILS import compute_tuning_colors


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
    vn = DF[DF["area"] != 1]
    vn_t = vn["t"].values
    vn_deg = normalize_angle(np.degrees(vn_t))
    total_vn = len(vn_deg)
    
    if total_vn == 0:
        return [1]

    start_angle, end_angle, sweep_width = get_v2_v4_range(vn_deg)

    step_size = 2.0
    front_plus = start_angle
    front_minus = end_angle
    
    counts = []
    max_steps = int(np.ceil(sweep_width / (2 * step_size))) + 2
    
    covered_mask = np.zeros(total_vn, dtype=bool)
    
    for i in range(max_steps):
        next_plus = normalize_angle(front_plus + step_size)
        next_minus = normalize_angle(front_minus - step_size)
        
        in1 = count_in_interval_mask(vn_deg, front_plus, next_plus)
        in2 = count_in_interval_mask(vn_deg, next_minus, front_minus)
        
        new_mask = (in1 | in2) & (~covered_mask)
        count = int(np.sum(new_mask))
        counts.append(count)
        
        covered_mask = covered_mask | new_mask
        
        front_plus = next_plus
        front_minus = next_minus
        
        if np.all(covered_mask):
             break


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"batch_size_{data_name}_{tag_name}.txt")
        with open(fname, "w") as f:
            for c in counts:
                f.write(f"{c}\n")
        print(f"Computed dynamic batch sizes (Total Vn={total_vn}, Sum={sum(counts)}). Saved to {fname}")
        
    return counts

def initDirectory(param, outputDir):
    """Placeholder output directory setup."""
    return outputDir


def computeV2V4MSE(DF, W):
    """Compute MSE between predicted and true V2-V4 tuning."""
    V1_count = len(DF[DF["area"] == 1])
    Vn_W = W[:, V1_count:]

    V1_df = DF[DF["area"] == 1].copy()
    V1_tuning = V1_df[["tuningX", "tuningY"]].values

    if hasattr(Vn_W, "cpu"):
        Vn_W = Vn_W.cpu().numpy()
    if hasattr(V1_tuning, "cpu"):
        V1_tuning = V1_tuning.cpu().numpy()

    predicted_tuning = Vn_W.T @ V1_tuning

    Vn_df = DF[DF["area"] != 1].copy()
    true_tuning = Vn_df[["tuningX", "tuningY"]].values

    mse = np.mean((predicted_tuning - true_tuning) ** 2)

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
    """Create and save a three-panel tuning comparison plot (true / predicted / eccentricity)."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    coords = DF[["x", "y"]].values
    areas = DF["area"].values.astype(int)

    from TUNING_COLOR_UTILS import get_tuning_colormap, should_flip_y_red_bottom
    cmap = get_tuning_colormap()
    true_colors_discrete = np.round(np.array(true_colors_array) * 10) / 10.0
    true_colors_discrete = np.clip(true_colors_discrete, 0.0, 1.0)
    pred_colors_discrete = np.round(np.array(pred_colors_array) * 10) / 10.0
    pred_colors_discrete = np.clip(pred_colors_discrete, 0.0, 1.0)
    true_rgba = [cmap(c) for c in true_colors_discrete]
    pred_rgba = [cmap(c) for c in pred_colors_discrete]

    v1_mask = (areas == 1)

    import re
    data_name = args.data
    is_rotated = bool(re.search(r'_(90|180|270)(_|$)', data_name))

    if is_rotated:
        if args.tag == "lh":
            flip_y = False
        elif args.tag == "rh":
            flip_y = True
        else:
            try:
                flip_y = should_flip_y_red_bottom(
                    coords[v1_mask] if np.any(v1_mask) else coords,
                    np.asarray(true_colors_discrete, dtype=float)[v1_mask] if np.any(v1_mask) else np.asarray(true_colors_discrete, dtype=float),
                )
            except Exception:
                flip_y = False
    else:
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

    if masked_v1_indices is None:
        masked_v1_set = set()
    else:
        masked_v1_set = set(masked_v1_indices)
    
    if unconnected_vn_indices is None:
        unconnected_vn_set = set()
    else:
        unconnected_vn_set = set(unconnected_vn_indices)
    
    is_center_idx = None
    if "is_center" in DF.columns:
        center_mask = DF["is_center"].values.astype(int) == 1
        if np.any(center_mask):
            is_center_idx = int(np.where(center_mask)[0][0])

    if distance_mode == "euclidean":
        out_base = os.path.join("../plot_mirror", "euclidean", "plots")
    else:
        out_base = os.path.join("../plot_mirror", "polar", "plots")
    os.makedirs(out_base, exist_ok=True)
    out_path = os.path.join(out_base, f"{args.data}_{args.tag}_tuning_compare{param_suffix}.png")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    marker_sym = 'o'
    marker_size = 10

    def plot_with_mask(ax, coords, areas, colors, title, is_predicted_panel=False):
        for a in np.unique(areas):
            idxs = np.where(areas == a)[0]
            if idxs.size == 0:
                continue
            
            # Separate masked and unmasked nodes for area 1
            if int(a) == 1:
                unmasked_idxs = [i for i in idxs if i not in masked_v1_set]
                masked_idxs = [i for i in idxs if i in masked_v1_set]

                if unmasked_idxs:
                    ax.scatter(
                        coords[unmasked_idxs, 0], coords[unmasked_idxs, 1],
                        c=[colors[i] for i in unmasked_idxs],
                        s=marker_size, alpha=1.0, linewidth=0,
                        marker=marker_sym,
                        label=f"Area {int(a)}"
                    )
                
                if masked_idxs:
                    ax.scatter(
                        coords[masked_idxs, 0], coords[masked_idxs, 1],
                        c=[colors[i] for i in masked_idxs],
                        s=marker_size, alpha=1.0, linewidth=0.5, edgecolors='gray',
                        marker=marker_sym,
                    )
            else:
                connected_idxs = [i for i in idxs if i not in unconnected_vn_set]
                unconnected_idxs = [i for i in idxs if i in unconnected_vn_set]
                
                if connected_idxs:
                    ax.scatter(
                        coords[connected_idxs, 0], coords[connected_idxs, 1],
                        c=[colors[i] for i in connected_idxs],
                        s=marker_size, alpha=1.0, linewidth=0,
                        marker=marker_sym,
                        label=f"Area {int(a)}"
                    )
                
                if unconnected_idxs:
                    if is_predicted_panel:
                        ax.scatter(
                            coords[unconnected_idxs, 0], coords[unconnected_idxs, 1],
                            c='black',
                            s=marker_size, alpha=1.0, linewidth=0,
                            marker=marker_sym,
                        )
                    else:
                        ax.scatter(
                            coords[unconnected_idxs, 0], coords[unconnected_idxs, 1],
                            c=[colors[i] for i in unconnected_idxs],
                            s=marker_size, alpha=1.0, linewidth=0,
                            marker=marker_sym,
                        )
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if is_center_idx is not None:
            ax.scatter(
                coords[is_center_idx, 0], coords[is_center_idx, 1],
                c='black',
                s=100, alpha=1.0, linewidth=2,
                marker='x',
                zorder=10
            )

    plot_with_mask(axes[0], coords, areas, true_rgba, "True", is_predicted_panel=False)
    plot_with_mask(axes[1], coords, areas, pred_rgba, "Polar angle", is_predicted_panel=True)


    try:
        if ref_ecc_colors is not None:
            pred_r = np.asarray(ref_ecc_colors, dtype=float)
        else:
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

    for ax in axes:
        if ax.get_legend():
            ax.legend().remove()

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Three-panel plot saved to {out_path}")


def loadDataDF(data="X1", tag="lh", mode="sphere"):
    """Load fMRI PKL and build the aligned DataFrame."""
    print(f"Loading fMRI data from pkl file: {data}_{tag} in {mode} mode...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device available: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    pkl_file = f"../data/{data}_{tag}.pkl"
    with open(pkl_file, "rb") as file:
        fMRI_data = pickle.load(file)

    print(f"Loaded {len(fMRI_data)} nodes from pkl file ({data}_{tag})")

    area = []
    x, y, z = [], [], []
    x_mds, y_mds = [], []
    tx, ty = [], []
    is_center_flags = []
    ID = []
    nodeIdx = []

    for item in fMRI_data:
        value = fMRI_data[item]
        nodeIdx.append(int(item))
        area.append(int(float(value["area"])))

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
            z.append(0.0)
        else:
            loc = value["loc_sphere"]
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])

        mds_loc = value["loc"]
        x_mds.append(mds_loc[0])
        y_mds.append(mds_loc[1])

        tx.append(float(value["tuning"][0]))
        ty.append(float(value["tuning"][1]))
        ID.append(int(item))
        is_center_flags.append(int(value.get("is_center", 0)))

    DF = pd.DataFrame(
        {
            "nodeIdx": nodeIdx,
            "ID": ID,
            "area": area,
            "x": x,
            "y": y,
            "z": z,
            "x_mds": x_mds,
            "y_mds": y_mds,
            "tuningX": tx,
            "tuningY": ty,
            "is_center": is_center_flags,
        }
    ).astype({"ID": int, "area": int})

    print(f"Created DataFrame with shape: {DF.shape}")
    print(f"Area distribution: {DF['area'].value_counts().sort_index().to_dict()}")
    print(f"Coordinate mode: {mode}")

    V1DF = DF[DF["area"] == 1].copy()
    VnDF = DF[DF["area"] != 1].copy()
    V1Count = V1DF.shape[0]
    VnCount = VnDF.shape[0]

    VnDF.loc[:, "V1Dist"] = 0.0
    VnDF.loc[:, "distGroup"] = pd.cut([0.0] * VnCount, 10)
    VnDF.loc[:, "anchorOrder"] = list(range(VnCount))

    V1DF.loc[:, "V1Dist"] = -1
    V1DF.loc[:, "distGroup"] = "(0)"
    V1DF.loc[:, "anchorOrder"] = -1

    temp = pd.concat([V1DF, VnDF])

    temp.loc[:, "tuningX"] = temp["tuningX"].astype(float)
    temp.loc[:, "tuningY"] = temp["tuningY"].astype(float)
    temp.loc[:, "tuningR"] = np.sqrt(np.square(temp["tuningX"]) + np.square(temp["tuningY"]))
    temp.loc[:, "tuningT"] = np.arctan2(temp["tuningY"], temp["tuningX"])

    center_mask = temp["is_center"].values.astype(int) == 1
    if not np.any(center_mask):
        raise RuntimeError("No is_center==1 node found in PKL; cannot set center.")
    center_idx = int(np.where(center_mask)[0][0])
    
    if mode == "mds":
        xs = temp["x"].values.astype(float)
        ys = temp["y"].values.astype(float)
        areas_arr = temp["area"].values.astype(int)
        xs_rot, ys_rot, _, _ = _rotate_to_align_x(xs, ys, areas_arr)

        mask_a1 = areas_arr == 1
        if np.any(mask_a1):
            cx = xs_rot[mask_a1].mean()
            cy = ys_rot[mask_a1].mean()
            cur_angle = np.arctan2(cy, cx)
            delta = np.pi - cur_angle
            xs_rot, ys_rot = _rotate_by_angle(xs_rot, ys_rot, delta)

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

    temp.loc[:, "r"] = np.sqrt(np.square(temp["x"]) + np.square(temp["y"]))
    temp.loc[:, "t"] = np.arctan2(temp["y"], temp["x"])

    temp.loc[:, "tuningTAlt"] = np.absolute(temp["t"]) * -1

    temp.loc[:, "Boundary"] = 0
    temp.loc[:, "Weight"] = 0

    tuningMin = np.min(np.absolute(temp["tuningT"]))
    temp.loc[((temp["area"] == 1) & (np.absolute(temp["tuningT"] - tuningMin) <= 5e-2)), "Boundary"] = 1
    temp.loc[temp["Boundary"] > 0, "Weight"] = 1

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
    """Save results and generate comparison plots. Returns (mse, pred_colors_array)."""
    import os
    import numpy as np
    import pandas as pd

    output_base_dir = "../results"
    if distance_mode == "euclidean":
        mode_dir = os.path.join(output_base_dir, "euclidean")
    else:
        mode_dir = os.path.join(output_base_dir, args.mode)
    os.makedirs(mode_dir, exist_ok=True)

    base_filename = f"{args.data}_{args.tag}_{args.algo}{param_suffix}"

    if hasattr(W, "cpu"):
        W_numpy = W.cpu().numpy()
    else:
        W_numpy = W

    V1_count = len(DF[DF["area"] == 1])
    Vn_W = W[:, V1_count:]

    V1_df = DF[DF["area"] == 1].copy()
    V1_tuning_vectors = V1_df[["tuningX", "tuningY"]].values

    if hasattr(Vn_W, "cpu"):
        Vn_W = Vn_W.cpu().numpy()
    if hasattr(V1_tuning_vectors, "cpu"):
        V1_tuning_vectors = V1_tuning_vectors.cpu().numpy()

    col_sums = np.sum(Vn_W, axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    Vn_W_norm = Vn_W / col_sums
    predicted_tuning = Vn_W_norm.T @ V1_tuning_vectors

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

    if tsv_only:
        print("Results saved (TSV-only):")
        print(f"   Tuning: {tuning_file}")
        return actual_params["mse"], None

    weight_file = os.path.join(mode_dir, f"W_{base_filename}.npz")
    save_dict = {"W": W_numpy}
    if node_generation_order is not None:
        node_order_np = np.array(node_generation_order, dtype=np.int32)
        save_dict["node_generation_order"] = node_order_np
    if batch_info is not None:
        import pickle
        batch_info_bytes = pickle.dumps(batch_info)
        save_dict["batch_info"] = np.array([batch_info_bytes], dtype=object)
    np.savez_compressed(weight_file, **save_dict)

    param_file = None
    pred_colors_array = None
    try:
        import matplotlib.pyplot as plt
        node_ids = DF["nodeIdx"].values.astype(int)

        true_map = {
            int(row["nodeIdx"]): np.array(
                [float(row["tuningX"]), float(row["tuningY"])], dtype=float
            )
            for _, row in DF.iterrows()
        }
        pred_map = {}
        for _, row in DF[DF["area"] == 1].iterrows():
            nid = int(row["nodeIdx"])
            pred_map[nid] = np.array(
                [float(row["tuningX"]), float(row["tuningY"])], dtype=float
            )
        for idx, (_, row) in enumerate(Vn_df.iterrows()):
            nid = int(row["nodeIdx"])
            pred_map[nid] = np.array(
                [float(predicted_tuning[idx, 0]), float(predicted_tuning[idx, 1])],
                dtype=float,
            )

        true_tuning_coords = np.array(
            [true_map.get(int(nid), [0.0, 0.0]) for nid in node_ids], dtype=float
        )
        pred_tuning_coords = np.array(
            [pred_map.get(int(nid), [0.0, 0.0]) for nid in node_ids], dtype=float
        )
        
        v1_mask = DF["area"].values == 1
        v1_indices = np.where(v1_mask)[0]

        if ref_colors_path is not None:
            ref = np.load(ref_colors_path, allow_pickle=True)
            ref_map = dict(zip(ref["nodeIdx"].astype(int), ref["colors"].astype(float)))
            available_bins = ref["available_bins"].astype(float)

            if "x_aligned" in ref and "y_aligned" in ref:
                ref_x_map = dict(zip(ref["nodeIdx"].astype(int), ref["x_aligned"].astype(float)))
                ref_y_map = dict(zip(ref["nodeIdx"].astype(int), ref["y_aligned"].astype(float)))
                for i, nid in enumerate(node_ids):
                    nid_int = int(nid)
                    if nid_int in ref_x_map:
                        DF.iloc[i, DF.columns.get_loc("x")] = ref_x_map[nid_int]
                        DF.iloc[i, DF.columns.get_loc("y")] = ref_y_map[nid_int]
                print("Overrode DF x,y with full-dataset aligned coordinates")

            ref_ecc_map = None
            if "eccentricity" in ref:
                ref_ecc_map = dict(zip(ref["nodeIdx"].astype(int), ref["eccentricity"].astype(float)))

            true_colors_array = np.array(
                [ref_map.get(int(nid), available_bins[len(available_bins)//2]) for nid in node_ids],
                dtype=float,
            )

            V1_colors_scalar = true_colors_array[v1_indices]
            V1_count_ref = len(v1_indices)
            Vn_W_ref = W_numpy[:, V1_count_ref:]
            col_sums_ref = np.sum(Vn_W_ref, axis=0, keepdims=True)
            col_sums_ref[col_sums_ref == 0] = 1.0
            Vn_W_norm_ref = Vn_W_ref / col_sums_ref
            pred_vn_colors = Vn_W_norm_ref.T @ V1_colors_scalar
            pred_vn_colors = np.round(pred_vn_colors * 10.0) / 10.0
            pred_vn_colors = np.clip(pred_vn_colors, available_bins.min(), available_bins.max())

            pred_colors_array = true_colors_array.copy()
            vn_indices = np.where(~v1_mask)[0]
            for i, df_idx in enumerate(vn_indices):
                pred_colors_array[df_idx] = pred_vn_colors[i]

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
            true_colors_array = compute_tuning_colors(true_tuning_coords, v1_mask=v1_mask, tag=args.tag)
            pred_colors_array = compute_tuning_colors(pred_tuning_coords, v1_mask=v1_mask, tag=args.tag)

            pred_colors_array[v1_indices] = true_colors_array[v1_indices]

        masked_v1_indices = []

        V1_count = len(v1_indices)
        Vn_W = W_numpy[:, V1_count:]
        col_sums = np.sum(Vn_W, axis=0)
        unconnected_vn_indices = []
        for idx, (_, row) in enumerate(Vn_df.iterrows()):
            if col_sums[idx] == 0:
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


