import os
import torch
import numpy as np
import pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from TUNING_COLOR_UTILS import compute_tuning_colors


def computeV2V4MSE(DF, W):
    """Mean squared error between predicted and measured V2-V4 tuning."""
    V1_count = len(DF[DF["area"] == 1])
    Vn_W = W[:, V1_count:]  # V1 to Vn connections

    V1_df = DF[DF["area"] == 1].copy()
    V1_tuning = V1_df[["tuningX", "tuningY"]].values

    if hasattr(Vn_W, "cpu"):  # Check if it's a PyTorch tensor
        Vn_W = Vn_W.cpu().numpy()
    if hasattr(V1_tuning, "cpu"):  # Check if it's a PyTorch tensor
        V1_tuning = V1_tuning.cpu().numpy()

    predicted_tuning = Vn_W.T @ V1_tuning  # (Vn_count, 2)

    Vn_df = DF[DF["area"] != 1].copy()
    true_tuning = Vn_df[["tuningX", "tuningY"]].values

    return float(np.mean((predicted_tuning - true_tuning) ** 2))


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

    from TUNING_COLOR_UTILS import get_tuning_colormap, should_flip_y_red_bottom
    cmap = get_tuning_colormap()
    true_colors_discrete = np.round(np.array(true_colors_array) * 10) / 10.0
    true_colors_discrete = np.clip(true_colors_discrete, 0.0, 1.0)
    pred_colors_discrete = np.round(np.array(pred_colors_array) * 10) / 10.0
    pred_colors_discrete = np.clip(pred_colors_discrete, 0.0, 1.0)
    true_rgba = [cmap(c) for c in true_colors_discrete]
    pred_rgba = [cmap(c) for c in pred_colors_discrete]

    v1_mask = (areas == 1)
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

    out_base = "../outputs/plots"
    os.makedirs(out_base, exist_ok=True)
    out_path = os.path.join(out_base, f"{args.data}_{args.tag}_tuning_compare{param_suffix}.png")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    marker_sym = 'o'
    marker_size = 10

    def plot_with_mask(ax, coords, areas, colors, title, is_predicted_panel=False):
        """
        is_predicted_panel: If True, show unconnected V2-V4 nodes in black (for predicted panel).
                           If False (for true tuning panel), show all nodes with their colors.
        """
        for a in np.unique(areas):
            idxs = np.where(areas == a)[0]
            if idxs.size == 0:
                continue
            
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
                zorder=10  # Ensure it's on top
            )

    plot_with_mask(axes[0], coords, areas, true_rgba, "True", is_predicted_panel=False)

    plot_with_mask(axes[1], coords, areas, pred_rgba, "Polar angle", is_predicted_panel=True)

    try:
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


def loadDataDF(data="NMT_gpr_grid", tag="lh"):
    """Load an input pkl and build the DataFrame the model runs on."""
    print(f"Loading data from pkl file: {data}_{tag}")

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

    with open(pkl_file, "rb") as file:
        fMRI_data = pickle.load(file)

    print(f"Loaded {len(fMRI_data)} nodes from pkl file ({data}_{tag})")

    area = []
    x, y = [], []
    sx, sy, sz = [], [], []          # sphere coordinates, when the pkl carries them
    tx, ty = [], []
    is_center_flags = []
    nodeIdx = []

    for item in fMRI_data:
        value = fMRI_data[item]
        nodeIdx.append(int(item))
        area.append(int(float(value["area"])))
        loc = value["loc"]                      # 2D MDS coordinates
        x.append(float(loc[0]))
        y.append(float(loc[1]))
        sph = value.get("loc_sphere", (0.0, 0.0, 0.0))
        sx.append(float(sph[0])); sy.append(float(sph[1])); sz.append(float(sph[2]))
        tx.append(float(value["tuning"][0]))
        ty.append(float(value["tuning"][1]))
        is_center_flags.append(int(value.get("is_center", 0)))

    DF = pd.DataFrame(
        {
            "nodeIdx": nodeIdx,
            "ID": nodeIdx,
            "area": area,
            "x": x,
            "y": y,
            "sx": sx,
            "sy": sy,
            "sz": sz,
            "tuningX": tx,
            "tuningY": ty,
            "is_center": is_center_flags,
        }
    ).astype({"ID": int, "area": int})

    print(f"Created DataFrame with shape: {DF.shape}")
    print(f"Area distribution: {DF['area'].value_counts().sort_index().to_dict()}")

    # Align the embedding: rotate the patch onto the x axis, put the V1 centroid
    # on the left, and move the origin to the foveal node.
    center_mask = DF["is_center"].values.astype(int) == 1
    if not np.any(center_mask):
        raise RuntimeError("No is_center==1 node found in pkl; cannot set center.")
    center_idx = int(np.where(center_mask)[0][0])

    xs = DF["x"].values.astype(float)
    ys = DF["y"].values.astype(float)
    areas_arr = DF["area"].values.astype(int)
    xs_rot, ys_rot, _, _ = _rotate_to_align_x(xs, ys, areas_arr)

    mask_a1 = areas_arr == 1
    if np.any(mask_a1):
        cur_angle = np.arctan2(ys_rot[mask_a1].mean(), xs_rot[mask_a1].mean())
        xs_rot, ys_rot = _rotate_by_angle(xs_rot, ys_rot, np.pi - cur_angle)

    c_x = float(xs_rot[center_idx])
    c_y = float(ys_rot[center_idx])
    DF.loc[:, "x"] = xs_rot - c_x
    DF.loc[:, "y"] = ys_rot - c_y
    print(f"Center (from is_center) for {data}_{tag}: ({c_x:.4f}, {c_y:.4f})")

    # Polar coordinates of the aligned embedding, used by the kernel.
    DF.loc[:, "r"] = np.sqrt(np.square(DF["x"]) + np.square(DF["y"]))
    DF.loc[:, "t"] = np.arctan2(DF["y"], DF["x"])

    # V1 nodes first, so that W = [I | 0] lines up with the node order.
    DF = DF.sort_values(by=["area", "ID"]).reset_index(drop=True)
    DF.loc[:, "ID"] = DF.index
    return DF


def save_baseline_results(
    DF,
    W,
    args,
    mse,
    param_suffix="",
    node_generation_order=None,
    batch_info=None,
    plot=True,
):
    """Write the predicted tuning (TSV) and the weight matrix (NPZ), and draw the
    comparison plot. Returns the predicted colors, which the animation reuses."""
    import os
    import numpy as np
    import pandas as pd

    mode_dir = "../outputs/predictions/mds"
    os.makedirs(mode_dir, exist_ok=True)
    base_filename = f"{args.data}_{args.tag}_deterministic{param_suffix}"

    if hasattr(W, "cpu"):  # Check if weight matrix is a PyTorch tensor
        W_numpy = W.cpu().numpy()
    else:
        W_numpy = W

    V1_count = len(DF[DF["area"] == 1])
    Vn_W = W[:, V1_count:]  # V1 to Vn connections

    V1_df = DF[DF["area"] == 1].copy()
    V1_tuning_vectors = V1_df[["tuningX", "tuningY"]].values

    if hasattr(Vn_W, "cpu"):
        Vn_W = Vn_W.cpu().numpy()
    if hasattr(V1_tuning_vectors, "cpu"):
        V1_tuning_vectors = V1_tuning_vectors.cpu().numpy()

    col_sums = np.sum(Vn_W, axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    Vn_W_norm = Vn_W / col_sums
    predicted_tuning = Vn_W_norm.T @ V1_tuning_vectors  # (Vn_count, 2)

    # Get true tuning for V2–V4
    Vn_df = DF[DF["area"] != 1].copy()
    true_tuning = Vn_df[["tuningX", "tuningY"]].values

    # Polar form of the same vectors, about the V1 anchor, and the node-wise
    # retinotopic error: these are the quantities reported in the paper.
    from TUNING_COLOR_UTILS import tuning_to_polar

    true_ecc, true_polar = tuning_to_polar(true_tuning, V1_tuning_vectors, args.tag)
    pred_ecc, pred_polar = tuning_to_polar(predicted_tuning, V1_tuning_vectors, args.tag)
    retinotopic_error = np.sqrt(np.sum((predicted_tuning - true_tuning) ** 2, axis=1))

    tuning_data = []
    for idx, (_, node) in enumerate(Vn_df.iterrows()):
        tuning_data.append(
            {
                "Node_ID": int(node["nodeIdx"]),
                "Area": int(node["area"]),
                "Pred_0": float(predicted_tuning[idx, 0]),
                "Pred_1": float(predicted_tuning[idx, 1]),
                "True_0": float(true_tuning[idx, 0]),
                "True_1": float(true_tuning[idx, 1]),
                "Pred_polar_angle_deg": float(pred_polar[idx]),
                "True_polar_angle_deg": float(true_polar[idx]),
                "Pred_eccentricity_deg": float(pred_ecc[idx]),
                "True_eccentricity_deg": float(true_ecc[idx]),
                "Retinotopic_error_deg": float(retinotopic_error[idx]),
            }
        )

    tuning_df = pd.DataFrame(tuning_data)
    tuning_file = os.path.join(mode_dir, f"predicted_{base_filename}.tsv")
    tuning_df.to_csv(tuning_file, sep="\t", index=False)


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

    pred_colors_array = None
    try:
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

        # compute_tuning_colors() defines the bin boundaries from V1 only, then
        # applies them to every node; V1 keeps its measured color in both panels.
        true_colors_array = compute_tuning_colors(true_tuning_coords, v1_mask=v1_mask, tag=args.tag)
        pred_colors_array = compute_tuning_colors(pred_tuning_coords, v1_mask=v1_mask, tag=args.tag)
        pred_colors_array[v1_indices] = true_colors_array[v1_indices]

        masked_v1_indices = []  # no V1 node is masked

        V1_count = len(v1_indices)
        Vn_W = W_numpy[:, V1_count:]  # V1->Vn connections
        col_sums = np.sum(Vn_W, axis=0)
        unconnected_vn_indices = []
        for idx, (_, row) in enumerate(Vn_df.iterrows()):
            vn_col_idx = idx  # Vn_df is already filtered to area != 1
            if col_sums[vn_col_idx] == 0:
                df_idx = DF[DF["nodeIdx"] == row["nodeIdx"]].index[0]
                unconnected_vn_indices.append(df_idx)

        if plot:
            plot_tuning_compare_two_panel(
                DF, true_colors_array, pred_colors_array, args,
                param_suffix=param_suffix,
                masked_v1_indices=masked_v1_indices if masked_v1_indices else None,
                unconnected_vn_indices=unconnected_vn_indices if unconnected_vn_indices else None,
                pred_tuning_coords=pred_tuning_coords,
            )
    except Exception as e:
        print(f"Warning: two-panel plot failed: {e}")

    print("Results saved:")
    print(f"   Tuning: {tuning_file}")
    print(f"   Weights: {weight_file}")
    return pred_colors_array




