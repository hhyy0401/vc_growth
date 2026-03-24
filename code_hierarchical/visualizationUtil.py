import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import torch
import pickle

def getColorMap():
    colors = ["green", "aqua", "pink", "red"]
    positions = np.linspace(0, 1, len(colors))
    custom_map = plt.cm.colors.LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)))
    return custom_map

def getTuningColor(DF, W, bin, absolute):
    adjDF = scaleColor(DF, bin=bin, absolute=absolute)
    V1DF = adjDF[adjDF["area"] == 1]
    VnDF = adjDF[adjDF["area"] != 1]
    V1Color = V1DF["color"].to_numpy()
    VnColor = (W[:,  W.shape[0]:].T @ V1Color).flatten()
    return V1Color, VnColor

def scaleColor(DF, tuning = "tuningT", bin=True, absolute=True):
    if absolute:
        DF.loc[:, "adjTuningT"] = np.abs(DF[tuning])
    else:
        DF.loc[:, "adjTuningT"] = DF[tuning]
    c_max = np.max(DF["adjTuningT"])
    c_min = np.min(DF["adjTuningT"])
    if bin:
        DF.loc[:, "color"] = 1.0
        DF.loc[(DF["adjTuningT"] - c_min) / (c_max - c_min) <= 0.75, "color"] = 0.67
        DF.loc[(DF["adjTuningT"] - c_min) / (c_max - c_min) <= 0.5, "color"] = 0.33
        DF.loc[(DF["adjTuningT"] - c_min) / (c_max - c_min) <= 0.25, "color"] = 0.0
    else:
        DF.loc[:, "color"] = (DF["adjTuningT"] - c_min) / (c_max - c_min)
    return DF

def color(dataDF, mseDF, W, recordDir, idx, scale=1, load=False): #Here, for the mirror reversal plot
    if load:
        record = np.load(os.path.join(recordDir, "W.npz"))
        W = record["W"]

    V1Data = dataDF[dataDF["area"] == 1]
    VnData = dataDF[dataDF["area"] != 1].reset_index()
    DF = pd.merge(dataDF, mseDF[["ID","EstBoundary", "estTuningT", "mse", "diffT"]], on="ID")
    borderDF = DF[DF["EstBoundary"] > 0]
    custom_colormap = getColorMap()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()
    V1Tuning, VnTuning = getTuningColor(dataDF, W, True, True)
    print('VnTuning unique (rounded 3):', np.unique(np.round(VnTuning, 3))[:20], '...')
    print('VnTuning unique count:', len(np.unique(np.round(VnTuning, 3))))
    cs = ax.scatter(V1Data["x_mds"], V1Data["y_mds"], c=V1Tuning, alpha=0.6, cmap=custom_colormap, marker="o", s=10)
    ax.scatter(VnData["x_mds"], VnData["y_mds"], c=VnTuning, alpha=1, cmap=custom_colormap, marker="v", s=10)
    ax.scatter(borderDF["x_mds"], borderDF["y_mds"], c="black", alpha=1, marker="+", s=20)

    plt.colorbar(cs)
    plt.savefig(os.path.join(recordDir, "W_BA_%d.png" % idx))
    plt.close()
    
def progressVis(dataDF, mseDF, W, recordDir, scale=1, load=False):
    if load:
        record = np.load(os.path.join(recordDir, "record.npz"))
        W = record["W"]
    custom_colormap = getColorMap()
    V1Count, VnCount, iter = W.shape
    V1DF = dataDF[dataDF["area"] == 1]
    VnDF = dataDF[dataDF["area"] != 1].reset_index()

    fig = plt.figure()
    ax = fig.gca()
    def animate(idx):
        ax.cla()
        temp = np.hstack((np.identity(V1Count), W[:, :, idx]))
        V1Tuning, VnTuning = getTuningColor(dataDF, temp, True, True)
        offset = np.argwhere(np.max(temp, axis=0) == 0).flatten()
        cs = ax.scatter(V1DF["x_mds"], V1DF["y_mds"], c=V1Tuning, alpha=0.6, cmap=custom_colormap, marker="o", s=10)
        ax.scatter(VnDF["x_mds"], VnDF["y_mds"], c=VnTuning, alpha=1, cmap=custom_colormap, marker="v", s=10)
        ax.scatter(VnDF["x_mds"].iloc[offset-V1Count], VnDF["y_mds"].iloc[offset-V1Count], color="black", alpha=1, marker="v", s=10)

    ani = animation.FuncAnimation(fig, animate, frames=iter, interval=200)
    writer = animation.FFMpegWriter(fps=20)
    ani.save(os.path.join(recordDir, "animation.mp4"), writer=writer)
    plt.close()

    color(dataDF, mseDF, np.hstack((np.identity(V1Count), W[:, :, -1])), recordDir, scale=1, load=False)

def drawTuning(data, outputDir):
    V1Data = data[data["area"] == 1]
    VnData = data[data["area"] != 1]
    borderDF = data[data["Boundary"] > 0]
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    custom_colormap = plt.cm.get_cmap('hsv')
    V1Tuning = V1Data["tuningT"]
    VnTuning = VnData["tuningT"]
    ax = axes
    cs = ax.scatter(V1Data["x_mds"], V1Data["y_mds"], c=V1Tuning, alpha=0.6, cmap=custom_colormap, marker="o", s=10)
    ax.scatter(VnData["x_mds"], VnData["y_mds"], c=VnTuning, alpha=1, cmap=custom_colormap, marker="v", s=10)
    ax.scatter(borderDF["x_mds"], borderDF["y_mds"], c="black", alpha=1, marker="+", s=20)
    plt.colorbar(cs)
    ax.set_xlabel('x_mds')
    ax.set_ylabel('y_mds')
    ax.set_title('tuningT')
    plt.tight_layout()
    plt.savefig('%s/tuning.png' % outputDir)

def drawLocation(data, outputDir):
    V1Data = data[data["area"] == 1]
    VnData = data[data["area"] != 1]
    borderDF = data[data["Boundary"] > 0]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    custom_colormap = plt.cm.get_cmap('hsv')
    ax = axes
    V1DF = data[data["area"] == 1]
    V1DF["color"] = np.absolute(V1DF["t"])
    cs = ax.scatter(V1DF["x_mds"], V1DF["y_mds"], c=V1DF["color"], alpha=0.6, cmap=custom_colormap, marker="v", s=10)
    selectedIdx = [250, 600, 850]
    for idx in selectedIdx:
        x, y = data.iloc[idx]["x_mds"], data.iloc[idx]["y_mds"]
        circle = patches.Circle((x, y), radius=1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(circle)

    plt.colorbar(cs)
    ax.scatter(borderDF["x_mds"], borderDF["y_mds"], c="black", alpha=1, marker="+", s=20)

    ax.set_xlabel('x_mds')
    ax.set_ylabel('y_mds')
    ax.set_title('t')
    plt.tight_layout()
    plt.savefig('%s/location.png' % outputDir)

def drawOrder(data, outputDir, offset):
    V1Data = data[data["area"] == 1]
    VnData = data[data["area"] != 1]
    V1Count = V1Data.shape[0]
    borderDF = data[data["Boundary"] > 0]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    
    custom_colormap = plt.cm.get_cmap('hsv')
    V1Tuning = V1Data["tuningT"]
    VnTuning = VnData["tuningT"]
    V1DF = data[data["area"] == 1]
    VnDF = data[data["area"] != 1]

    ax = axes
    cs = ax.scatter(V1DF["x_mds"], V1DF["y_mds"], c=V1Tuning, alpha=0.6, cmap=custom_colormap, marker="o", s=10)
    ax.scatter(VnDF["x_mds"], VnDF["y_mds"], c=VnTuning, alpha=1, cmap=custom_colormap, marker="v", s=10)
    ax.scatter(VnDF["x_mds"].iloc[offset-V1Count], VnDF["y_mds"].iloc[offset-V1Count], color="black", alpha=1, marker="v", s=10)
    plt.colorbar(cs)
    ax.set_xlabel('x_mds')
    ax.set_ylabel('y_mds')
    ax.set_title('tuningT')
    plt.tight_layout()
    plt.savefig('%s/order_%d.png' % (outputDir, offset))

def visualizeProportion(data, outputDir):
    proportionDF = pd.read_csv("%s/proportion.csv" % outputDir)
    mseDf = pd.read_csv("%s/mse.csv" % outputDir)
    print(f"Average MSE: {mseDf['mse'].mean()}")
    V1DF = data[data["area"] == 1]
    VnDF = data[data["area"] != 1]
    V1Count = V1DF.shape[0]
    

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    custom_colormap = plt.cm.get_cmap('winter')
    
    ax = axes[0]
    proportion = proportionDF.iloc[:, 1:]
    proportion = proportion.sum(axis=1) / proportion.shape[1]
    cs = ax.scatter(VnDF["x_mds"], VnDF["y_mds"], c=proportion, alpha=0.6, cmap="winter", marker="v", s=10)
    plt.colorbar(cs, ax=ax)
    ax.set_title('Proportion')

    ax = axes[1]
    avgDiff = proportionDF.iloc[:, 1:]
    avgDiff = np.square(avgDiff.sub(avgDiff.mean(axis=1), axis=0)).mean(axis=1)
    cs = ax.scatter(VnDF["x_mds"], VnDF["y_mds"], c=avgDiff, alpha=0.6, cmap="winter", marker="v", s=10)
    plt.colorbar(cs, ax=ax)
    ax.set_title('Average Difference')

    plt.tight_layout()
    plt.savefig('%s/proportion.png' % outputDir)

from utils import loadDataDF  # reuse canonical implementation


def create_video_animation(
    data,
    tag,
    mode,
    alpha=1.0,
    euclidean=6.0,
    tangent=30.0,
    DF=None,
    matrix=None,
    pred_colors_array=None,
    distance_mode="polar",
    custom_batch_mode=None,
):
    """Create an HTML animation showing node activation order."""
    import plotly.graph_objects as go
    from plotly.offline import plot
    from TUNING_COLOR_UTILS import (
        get_tuning_colormap,
        round_color_bins,
        should_flip_y_red_bottom,
        compute_tuning_colors,
    )

    print("Creating video animation from saved simulation results…")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0,1] (got {alpha})")
    print(f"Parameters: alpha={float(alpha):.1f}, euclidean={euclidean}, tangent={tangent}")

    # DF, matrix, and pred_colors_array must be provided
    if DF is None or matrix is None:
        raise ValueError("DF and matrix must be provided (from runSimulation). No re-simulation will be performed.")
    if pred_colors_array is None:
        raise ValueError("pred_colors_array must be provided (from save_baseline_results). No re-computation will be performed.")
    
    node_gen_order = list(matrix.node_generation_order)
    print(f"Tracked V2–V4 generation for {len(node_gen_order)} nodes")
    print(f"Using saved batch_info with {len(matrix.batch_info)} batches")
    print(f"Using saved pred_colors_array with {len(pred_colors_array)} colors")
    
    if hasattr(matrix.matrixW, "cpu"):
        W_np = matrix.matrixW.cpu().numpy()
    else:
        W_np = matrix.matrixW
    
    coords = DF[["x", "y"]].values.astype(float)
    all_x, all_y = coords[:, 0], coords[:, 1]
    v1_mask = (DF["area"].values.astype(int) == 1)
    V1_df = DF[DF["area"] == 1].copy()
    Vn_df = DF[DF["area"] != 1].copy().reset_index(drop=True)
    V1_count = len(V1_df)
    V1_nodes = V1_df["ID"].values
    
    node_id_to_df_idx = {int(node_id): idx for idx, node_id in enumerate(DF["ID"].values)}
    
    try:
        true_tuning_coords = DF[["tuningX", "tuningY"]].values.astype(float)
        true_colors = compute_tuning_colors(true_tuning_coords, v1_mask=v1_mask, tag=tag)
        true_bins = round_color_bins(true_colors)
        flip_y = should_flip_y_red_bottom(
            np.stack([all_x[v1_mask], all_y[v1_mask]], axis=1) if np.any(v1_mask) else np.stack([all_x, all_y], axis=1),
            true_bins[v1_mask] if np.any(v1_mask) else true_bins,
        )
    except Exception:
        flip_y = False
    if flip_y:
        all_y = all_y.copy()
        all_y *= -1.0

    cmap = get_tuning_colormap()
    pred_bins = round_color_bins(np.asarray(pred_colors_array, dtype=float))
    all_colors = []
    for c in pred_bins:
        rgba = cmap(float(c))
        all_colors.append(f"rgb({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)})")
    
    node_groups = []
    node_groups.append(
        {
            "timestamp": 0,
            "nodes": list(V1_nodes),
            "title": "V1 nodes only",
        }
    )

    MIN_NODES_PER_FRAME = 20
    current_frame_nodes = []
    
    batch_info = matrix.batch_info
    if batch_info:
        for batch_idx, batch_nodes_info in enumerate(batch_info, start=1):
            batch_vn_ids = []
            for vn_idx, v1_indices, pred_tuning_color in batch_nodes_info:
                if 0 <= vn_idx < len(Vn_df):
                    batch_vn_ids.append(int(Vn_df.iloc[vn_idx]["ID"]))
            
            current_frame_nodes.extend(batch_nodes_info)

            current_frame_vn_ids = []
            for b_vn_idx, _, _ in current_frame_nodes:
                if 0 <= b_vn_idx < len(Vn_df):
                    current_frame_vn_ids.append(int(Vn_df.iloc[b_vn_idx]["ID"]))
            
            if len(current_frame_vn_ids) >= MIN_NODES_PER_FRAME or batch_idx == len(batch_info):
                frame_idx = len(node_groups)
                node_groups.append(
                    {
                        "timestamp": frame_idx,
                        "nodes": current_frame_vn_ids,
                        "title": f"V2–V4 frame {frame_idx} (added {len(current_frame_vn_ids)} nodes)",
                        "batch_info": current_frame_nodes,
                    }
                )
                current_frame_nodes = []

    frames = []
    all_visible_nodes = set()

    for g_idx, group in enumerate(node_groups):
        is_last_frame = (g_idx == len(node_groups) - 1)
        all_visible_nodes.update(group["nodes"])

        frame_colors = ["rgb(220, 220, 220)"] * len(DF)
        frame_opacity = [0.1] * len(DF)
        # Base size for all nodes
        frame_size = [8] * len(DF)

        visible_vn_ids = [nid for nid in all_visible_nodes if nid not in V1_nodes]
        current_group_vn_ids = [nid for nid in group["nodes"] if nid not in V1_nodes]
        
        parent_mask = np.zeros(V1_count, dtype=bool)
        if "batch_info" in group:
            batch_nodes_info = group["batch_info"]
            for vn_idx, v1_indices, pred_tuning_color in batch_nodes_info:
                for v1_idx in v1_indices:
                    if 0 <= v1_idx < V1_count:
                        parent_mask[v1_idx] = True
        for node_id in V1_nodes:
            df_idx = node_id_to_df_idx[int(node_id)]
            v1_idx = df_idx
            frame_colors[df_idx] = all_colors[df_idx]
            frame_opacity[df_idx] = 1.0
            if (not is_last_frame) and 0 <= v1_idx < len(parent_mask) and parent_mask[v1_idx]:
                frame_size[df_idx] = 14
        vn_has_connections = {}
        if "batch_info" in group:
            batch_nodes_info = group["batch_info"]
            for vn_idx, v1_indices, pred_tuning_color in batch_nodes_info:
                if 0 <= vn_idx < len(Vn_df):
                    vn_node_id = int(Vn_df.iloc[vn_idx]["ID"])
                    vn_has_connections[vn_node_id] = len(v1_indices) > 0
        
        for node_id in visible_vn_ids:
            df_idx = node_id_to_df_idx[int(node_id)]
            if node_id in vn_has_connections:
                if vn_has_connections[node_id]:
                    frame_colors[df_idx] = all_colors[df_idx]
                    frame_opacity[df_idx] = 1.0
                    # Increase size for nodes in current batch (but not in last frame)
                    if node_id in current_group_vn_ids and not is_last_frame:
                        frame_size[df_idx] = 14  # Larger size for newly added nodes
                else:
                    frame_colors[df_idx] = "rgb(0, 0, 0)"
                    frame_opacity[df_idx] = 1.0
            else:
                # For nodes not in current batch, check if they appeared in previous batches
                # (they should have connections if they're visible)
                frame_colors[df_idx] = all_colors[df_idx]
                frame_opacity[df_idx] = 1.0
        node_ids_all = [str(int(v)) for v in DF["ID"].values]
        frame_data = [
            go.Scatter(
                x=list(all_x),
                y=list(all_y),
                mode="markers",
                text=node_ids_all,
                hovertemplate="Node ID: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                marker=dict(
                    color=frame_colors,
                    size=frame_size,
                    opacity=frame_opacity,
                    symbol="circle",
                    line=dict(width=0),
                ),
                showlegend=False,
                name="Nodes",
            )
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(group["timestamp"]),
                layout=go.Layout(
                    title=f"t={group['timestamp']}: {group['title']}",
                ),
            )
        )
        
    node_ids_all = [str(int(v)) for v in DF["ID"].values]
    initial_data = [
        go.Scatter(
            x=list(all_x),
            y=list(all_y),
            mode="markers",
            text=node_ids_all,
            hovertemplate="Node ID: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            marker=dict(
                color=["rgb(220, 220, 220)"] * len(DF),
                size=8,
                opacity=0.2,
                symbol="circle",
                line=dict(width=0),
            ),
            showlegend=False,
            name="Nodes",
        )
    ]

    fig = go.Figure(
        data=initial_data,
        frames=frames,
        layout=go.Layout(
                    title="Node animation (order + predicted tuning color)",
            xaxis=dict(title="x", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="y", showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 800, "redraw": True}}],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}}],
                        },
                    ],
                }
            ],
        ),
    )

    if distance_mode == "euclidean":
        out_dir = os.path.join("..", "plot_mirror", "euclidean", "video")
    else:
        out_dir = os.path.join("..", "plot_mirror", "polar", "video")
    os.makedirs(out_dir, exist_ok=True)
    e_str = f"{euclidean:.2f}"
    a_str = f"{tangent:.2f}"
    alpha_str = f"{float(alpha):.1f}"
    # Always use .html extension (polar mode)
    filename_suffix = f"_{e_str}_{a_str}_{alpha_str}"
    if custom_batch_mode:
        filename_suffix += f"_{custom_batch_mode}"
    out_path = os.path.join(out_dir, f"{data}_{tag}_animation{filename_suffix}.html")
    plot(fig, filename=out_path, auto_open=False)
    print(f"Video animation saved to: {out_path}")

    # Generate timestamp plot for custom batch modes
    if custom_batch_mode:
        create_timestamp_plot(
            data, tag, alpha=alpha, euclidean=euclidean, tangent=tangent,
            all_x=all_x, all_y=all_y, v1_mask=v1_mask,
            V1_count=V1_count,
            node_generation_order=matrix.node_generation_order,
            custom_batch_mode=custom_batch_mode,
            batch_info=matrix.batch_info,
            areas=DF["area"].values,
        )


def create_timestamp_plot(
    data, tag, alpha, euclidean, tangent,
    all_x, all_y, v1_mask, V1_count,
    node_generation_order, custom_batch_mode,
    batch_info=None, areas=None,
):
    """Create a timestamp plot with batch-level coloring."""
    import colorsys
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    N = len(all_x)
    vn_mask = ~v1_mask

    # node_batch_map: vn_col -> (batch_id, position_in_batch, batch_size)
    node_batch_map = {}
    n_batches = 0
    if batch_info:
        n_batches = len(batch_info)
        for b_idx, batch in enumerate(batch_info):
            for pos, (vn_col, _, _) in enumerate(batch):
                node_batch_map[int(vn_col)] = (b_idx, pos, len(batch))

    colors = np.zeros((N, 4), dtype=float)

    # V1: light gray
    v1_color = (0.82, 0.82, 0.82, 1.0)
    colors[v1_mask] = v1_color

    # Build 30-color categorical palette from tab20 (20) + tab20b (10)
    _tab20 = plt.cm.tab20
    _tab20b = plt.cm.tab20b
    batch_palette = []
    for k in range(20):
        batch_palette.append(np.array(_tab20(k / 20.0)[:3]))
    for k in range(10):
        batch_palette.append(np.array(_tab20b(k / 20.0)[:3]))

    # Vn: batch-based coloring, no transparency (blend with white/black instead)
    # frac 0 (first in batch) = white-blended, frac 0.5 = normal, frac 1 = dark
    white = np.array([1.0, 1.0, 1.0])
    for i in range(N):
        if not vn_mask[i]:
            continue
        vn_col = i - V1_count
        if vn_col not in node_batch_map:
            colors[i] = (0.5, 0.5, 0.5, 1.0)
            continue

        b_id, pos, bsize = node_batch_map[vn_col]
        base_rgb = batch_palette[b_id % len(batch_palette)]
        if bsize > 1:
            frac = pos / (bsize - 1)
        else:
            frac = 0.5
        if frac <= 0.5:
            # white → normal: blend base with white
            t = frac / 0.5  # 0→1
            rgb = base_rgb * t + white * (1.0 - t)
        else:
            # normal → dark shadow
            t = (frac - 0.5) / 0.5  # 0→1
            dark_factor = 1.0 - 0.92 * t
            rgb = base_rgb * dark_factor
        colors[i] = (rgb[0], rgb[1], rgb[2], 1.0)

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_cb_area = fig.add_subplot(gs[0, 1])

    ax.set_aspect("equal")
    ax.axis("off")
    ax_cb_area.axis("off")

    marker_size = 36

    # Draw V1
    sc1 = ax.scatter(
        all_x[v1_mask], all_y[v1_mask],
        c=[v1_color], s=marker_size, marker="o",
        linewidths=0, edgecolors="none", rasterized=True, zorder=1,
    )
    sc1.set_antialiased(False)
    # Draw Vn
    sc2 = ax.scatter(
        all_x[vn_mask], all_y[vn_mask],
        c=colors[vn_mask], s=marker_size, marker="o",
        linewidths=0, edgecolors="none", rasterized=True, zorder=2,
    )
    sc2.set_antialiased(False)

    # Within-batch gradient colorbar
    # white → gray(mid) → black
    wb_cmap = LinearSegmentedColormap.from_list(
        "within_batch", [(1, 1, 1), (0.5, 0.5, 0.5), (0.08, 0.08, 0.08)], N=256
    )
    norm_wb = Normalize(vmin=0, vmax=1)
    sm_wb = plt.cm.ScalarMappable(norm=norm_wb, cmap=wb_cmap)
    sm_wb.set_array([])
    cax_wb = ax_cb_area.inset_axes([-0.35, 0.32, 0.22, 0.36])
    cbar_wb = fig.colorbar(sm_wb, cax=cax_wb, orientation="vertical")
    cbar_wb.set_ticks([0, 1])
    cbar_wb.set_ticklabels(["0", "1"])
    # Increase within-batch colorbar tick label size (2x)
    cbar_wb.ax.tick_params(labelsize=26)

    # Batch index colorbar
    n_colors = max(n_batches, 1)
    cb_colors_list = [tuple(batch_palette[b % len(batch_palette)]) for b in range(n_colors)]
    batch_cmap = LinearSegmentedColormap.from_list("batch", cb_colors_list, N=n_colors)

    norm = Normalize(vmin=1, vmax=n_colors)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=batch_cmap)
    sm.set_array([])

    cax = ax_cb_area.inset_axes([0.44, 0.25, 0.38, 0.5])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    tick_vals = [1, n_colors // 2, n_colors]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([str(v) for v in tick_vals])
    # Increase batch-index colorbar tick label size (2x)
    cbar.ax.tick_params(labelsize=32)

    # Save
    out_dir = os.path.join("..", "plot_mirror", "polar", "plots", "timestamp")
    os.makedirs(out_dir, exist_ok=True)
    e_str = f"{euclidean:.2f}"
    a_str = f"{tangent:.2f}"
    alpha_str = f"{float(alpha):.1f}"
    fname = f"{data}_{tag}_{e_str}_{a_str}_{alpha_str}_{custom_batch_mode}.png"
    out_path = os.path.join(out_dir, fname)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Timestamp plot saved to: {out_path}")