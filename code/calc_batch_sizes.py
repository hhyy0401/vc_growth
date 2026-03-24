"""Compute angular-sweep batch-size schedules for each (data, tag) combination."""

import os
import sys
import numpy as np
import argparse
from utils import loadDataDF


def normalize_angle(a):
    """Normalize angle to [0, 360)."""
    return a % 360


def get_v1_boundaries_robust(deg, fraction=0.95):
    """Find V1 block boundaries via shortest arc containing `fraction` of points."""
    n = len(deg)
    k = int(np.ceil(fraction * n))

    deg_sorted = np.sort(deg)
    deg_extended = np.concatenate([deg_sorted, deg_sorted + 360])

    widths = deg_extended[k-1:n+k-1] - deg_extended[:n]
    min_width_idx = np.argmin(widths)
    min_width = widths[min_width_idx]
    
    start_angle = normalize_angle(deg_extended[min_width_idx])
    end_angle = normalize_angle(deg_extended[min_width_idx + k - 1])
    
    gap_width = 360.0 - min_width
    boundary_upper = end_angle
    boundary_lower = start_angle

    return boundary_upper, boundary_lower, gap_width


def count_in_interval(deg, start, end):
    """Count nodes whose angle is within (start, end], handling wrap-around."""
    s = normalize_angle(start)
    e = normalize_angle(end)
    if s <= e:
        return int(np.sum((deg > s) & (deg <= e)))
    else:
        return int(np.sum((deg > s) | (deg <= e)))


def generate_batch_sizes(data, tag, output_dir="../batch_size", do_plot=False):
    print(f"\n{'='*50}")
    print(f"Generating batch sizes for {data} {tag}...")

    try:
        df = loadDataDF(data, tag, "mds")
    except Exception as e:
        print(f"Skipping {data} {tag}: {e}")
        return

    v1 = df[df["area"] == 1]
    v1_t = v1["t"].values
    v1_deg = normalize_angle(np.degrees(v1_t))

    boundary_upper, boundary_lower, gap_width = get_v1_boundaries_robust(v1_deg, fraction=0.95)
    
    v1_width = 360 - gap_width
    print(f"V1 span (95% core) : {v1_width:.1f}°")
    print(f"Gap span (sweep)   : {gap_width:.1f}°")
    print(f"Boundary upper (sweep +): {boundary_upper:.2f}°")
    print(f"Boundary lower (sweep -): {boundary_lower:.2f}°")

    vn = df[df["area"] != 1]
    vn_t = vn["t"].values
    vn_deg = normalize_angle(np.degrees(vn_t))
    total_vn = len(vn_deg)
    print(f"Total Vn nodes: {total_vn}")

    step_size = 2.0
    front_plus  = boundary_upper
    front_minus = boundary_lower

    counts = []
    max_steps = int(np.ceil(gap_width / (2 * step_size))) + 5

    remaining_dist = gap_width

    for i in range(max_steps):
        next_plus = normalize_angle(front_plus + step_size)
        next_minus = normalize_angle(front_minus - step_size)
        
        c1 = count_in_interval(vn_deg, front_plus, next_plus)
        c2 = count_in_interval(vn_deg, next_minus, front_minus)
        
        counts.append(c1 + c2)
        
        front_plus = next_plus
        front_minus = next_minus
        
        if (i + 1) * 2 * step_size >= gap_width:
             print(f"Fronts met/crossed at step {i + 1}")
             break

    total_swept = sum(counts)
    print(f"Sum of batch sizes: {total_swept}")
    print(f"Coverage: {total_swept / total_vn * 100:.1f}% of Vn nodes")

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"batch_size_{data}_{tag}.txt")
    with open(filename, "w") as f:
        for c in counts:
            f.write(f"{c}\n")
    print(f"Saved {len(counts)} steps to {filename}")

    if do_plot:
        _plot_sweep(df, v1_deg, vn_deg, boundary_upper, boundary_lower,
                    gap_width, step_size, counts, data, tag, output_dir)


def _plot_sweep(df, v1_deg, vn_deg, boundary_upper, boundary_lower,
                gap_width, step_size, counts, data, tag, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    all_x = df["x"].values
    all_y = df["y"].values
    areas = df["area"].values
    all_t = df["t"].values
    all_deg = normalize_angle(np.degrees(all_t))

    vn_mask = areas != 1
    vn_x = all_x[vn_mask]
    vn_y = all_y[vn_mask]
    vn_angles = all_deg[vn_mask]

    node_step = np.full(len(vn_angles), -1, dtype=int)
    fp = boundary_upper
    fm = boundary_lower
    actual_steps = len(counts)
    
    for s in range(actual_steps):
        np_ = normalize_angle(fp + step_size)
        nm = normalize_angle(fm - step_size)
        
        in1 = _in_interval_mask(vn_angles, fp, np_)
        in2 = _in_interval_mask(vn_angles, nm, fm)
        
        mask = (in1 | in2) & (node_step == -1)
        node_step[mask] = s
        
        fp = np_
        fm = nm

    v1_x = all_x[~vn_mask]
    v1_y = all_y[~vn_mask]
    ax.scatter(v1_x, v1_y, c="gray", s=8, alpha=0.3, label="V1")

    swept = node_step >= 0
    if np.any(swept):
        sc = ax.scatter(vn_x[swept], vn_y[swept], c=node_step[swept],
                        cmap="plasma", s=15, alpha=0.9)
        plt.colorbar(sc, ax=ax, label="Step Index")

    if np.any(~swept):
        ax.scatter(vn_x[~swept], vn_y[~swept], c="white", s=8, alpha=0.15, label="Unswept Vn")

    ax.set_title(f"Sweep: {data} {tag}\nSum={sum(counts)}, TotalVn={len(vn_angles)}", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    ax2 = axes[1]
    ax2.bar(range(len(counts)), counts, color="cyan", alpha=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Batch Size")
    ax2.set_title("Batch Size Schedule")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"sweep_plot_{data}_{tag}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sweep plot to {plot_path}")


def _in_interval_mask(deg, start, end):
    s = normalize_angle(start)
    e = normalize_angle(end)
    if s <= e:
        return (deg > s) & (deg <= e)
    else:
        return (deg > s) | (deg <= e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute angular-sweep batch-size schedules")
    parser.add_argument("--data", nargs="+",
                        default=["R1_gpr_grid", "S1_gpr_grid", "S2_gpr_grid",
                                 "S3_gpr_grid", "S4_gpr_grid", "S5_gpr_grid", "S6_gpr_grid"])
    parser.add_argument("--tags", nargs="+", default=["lh", "rh"])
    parser.add_argument("--plot", action="store_true", help="Generate verification plots")
    args = parser.parse_args()

    for d in args.data:
        for t in args.tags:
            generate_batch_sizes(d, t, do_plot=args.plot)
