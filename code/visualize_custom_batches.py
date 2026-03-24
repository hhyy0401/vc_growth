"""
Standalone visualization of custom batch assignments.

Usage:
    python visualize_custom_batches.py --data R1_gpr_grid --tag lh --mode mds
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import loadDataDF
from custom_batch import (
    compute_v1_boundary,
    compute_batches_angle,
    compute_batches_polar,
    compute_batches_euclidean,
    compute_batches_x,
    get_custom_node_order,
    _boundary_line_coords,
)


def visualize_batches(DF, batches, line1, line2, center_xy, center_idx, mode_name, save_path):
    """Plot nodes colored by batch assignment with boundary lines."""
    coords = DF[["x", "y"]].values.astype(float)
    areas = DF["area"].values.astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    v1_mask = areas == 1
    ax.scatter(coords[v1_mask, 0], coords[v1_mask, 1],
               c="lightgray", s=8, alpha=0.5, label="V1", zorder=1)

    n_batches = max(batches.keys()) + 1 if batches else 1
    cmap = cm.get_cmap("tab20", n_batches)

    for batch_id, node_indices in sorted(batches.items()):
        if len(node_indices) == 0:
            continue
        color = cmap(batch_id % 20)
        ax.scatter(coords[node_indices, 0], coords[node_indices, 1],
                   c=[color], s=12, alpha=0.8, label=f"Batch {batch_id}", zorder=2)

    bnd1_coords = _boundary_line_coords(DF, line1)
    bnd2_coords = _boundary_line_coords(DF, line2)

    if len(bnd1_coords) > 0:
        full_line1 = np.vstack([center_xy, bnd1_coords])
        ax.plot(full_line1[:, 0], full_line1[:, 1], "r-", linewidth=2, label="Boundary 1", zorder=5)
    if len(bnd2_coords) > 0:
        full_line2 = np.vstack([center_xy, bnd2_coords])
        ax.plot(full_line2[:, 0], full_line2[:, 1], "b-", linewidth=2, label="Boundary 2", zorder=5)

    ax.scatter([center_xy[0]], [center_xy[1]], c="black", s=200,
               marker="x", linewidths=3, zorder=10, label="Center")

    ax.set_title(f"Custom Batches: {mode_name} ({len(batches)} non-empty batches)")
    ax.set_aspect("equal")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def visualize_node_order(DF, ordered_nodes, line1, line2, center_xy, mode_name, save_path):
    """Plot nodes colored by visitation order (gradient from early=blue to late=red)."""
    coords = DF[["x", "y"]].values.astype(float)
    areas = DF["area"].values.astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    v1_mask = areas == 1
    ax.scatter(coords[v1_mask, 0], coords[v1_mask, 1],
               c="lightgray", s=8, alpha=0.5, zorder=1)

    n_nodes = len(ordered_nodes)
    if n_nodes > 0:
        order_colors = np.linspace(0, 1, n_nodes)
        cmap = cm.get_cmap("coolwarm")
        for i, idx in enumerate(ordered_nodes):
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=[cmap(order_colors[i])], s=12, alpha=0.8, zorder=2)

    bnd1_coords = _boundary_line_coords(DF, line1)
    bnd2_coords = _boundary_line_coords(DF, line2)
    if len(bnd1_coords) > 0:
        full_line1 = np.vstack([center_xy, bnd1_coords])
        ax.plot(full_line1[:, 0], full_line1[:, 1], "r-", linewidth=2, zorder=5)
    if len(bnd2_coords) > 0:
        full_line2 = np.vstack([center_xy, bnd2_coords])
        ax.plot(full_line2[:, 0], full_line2[:, 1], "b-", linewidth=2, zorder=5)

    ax.scatter([center_xy[0]], [center_xy[1]], c="black", s=200,
               marker="x", linewidths=3, zorder=10)

    ax.set_title(f"Node Order: {mode_name} (blue=early, red=late)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize custom batch assignments")
    parser.add_argument("--data", type=str, default="R1_gpr_grid")
    parser.add_argument("--tag", type=str, default="lh")
    parser.add_argument("--mode", type=str, default="mds")
    parser.add_argument("--n_batches", type=int, default=30)
    args = parser.parse_args()

    DF = loadDataDF(args.data, args.tag, args.mode)
    line1, line2, center_xy, center_idx = compute_v1_boundary(DF)
    print(f"\nBoundary line 1: {len(line1)} nodes")
    print(f"Boundary line 2: {len(line2)} nodes")
    print(f"Center: idx={center_idx}, xy=({center_xy[0]:.4f}, {center_xy[1]:.4f})")

    out_dir = f"../plot_custom_batches/{args.mode}"
    os.makedirs(out_dir, exist_ok=True)

    mode_funcs = {
        "angle": compute_batches_angle,
        "polar": compute_batches_polar,
        "euclidean": compute_batches_euclidean,
        "x": compute_batches_x,
    }

    for mode_name, func in mode_funcs.items():
        print(f"\n--- {mode_name} ---")
        batches = func(DF, line1, line2, center_xy, center_idx, args.n_batches)
        total_nodes = sum(len(v) for v in batches.values())
        print(f"  Non-empty batches: {len(batches)}")
        print(f"  Total assigned nodes: {total_nodes}")
        for b_id in sorted(batches.keys()):
            print(f"  Batch {b_id}: {len(batches[b_id])} nodes")

        save_path = os.path.join(out_dir, f"batches_{mode_name}_{args.data}_{args.tag}.png")
        visualize_batches(DF, batches, line1, line2, center_xy, center_idx, mode_name, save_path)

        for ecc in ["up", "down"]:
            full_mode = f"custom_{mode_name}_{ecc}"
            ordered_nodes, ordered_batches = get_custom_node_order(DF, mode_name, ecc, args.n_batches)
            order_path = os.path.join(out_dir, f"order_{mode_name}_{ecc}_{args.data}_{args.tag}.png")
            visualize_node_order(DF, ordered_nodes, line1, line2, center_xy, full_mode, order_path)

    print(f"\nAll visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
