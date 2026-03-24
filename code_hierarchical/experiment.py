import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy version.*")

from polarModel import VisualMatrix3D
from visualizationUtil import create_video_animation
from scipy.optimize import dual_annealing
import argparse
import os
import numpy as np
import pandas as pd
import sys
from utils import loadDataDF, computeV2V4MSE, save_baseline_results

# Try to import skopt for TPE optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    print("skopt not available, falling back to dual_annealing")
    SKOPT_AVAILABLE = False


def runSimulation(args):
    DF = loadDataDF(args.data, args.tag, args.mode)

    distance_mode = getattr(args, "distance_mode", "polar")
    print(f"Using {distance_mode} distance mode")

    eff_num_degree = 1
    alpha = float(getattr(args, "alpha", 1.0))
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0,1] (got {alpha})")

    radius_val = float(args.radius)
    tangent_val = float(args.tangent)

    param_mode = "fit" if args.sim_mode == "fit" else "record"
    sample_matrix = -1 if args.algo == "deterministic" else 1

    hierarchical = bool(getattr(args, "hierarchical", False))
    stage_ratio = float(getattr(args, "stage_ratio", 0.90))
    in_degree_max = int(getattr(args, "in_degree_max", 1))
    max_stages = int(getattr(args, "max_stages", 50))

    param_dict = {
        "mode": param_mode,
        "coordinate_mode": args.mode,
        "num_degree": eff_num_degree,
        "alpha": alpha,
        "radius": radius_val,
        "tangent": tangent_val,
        "distance_mode": distance_mode,
        "batch_size_start": int(args.batch_size_start),
        "batch_size_end": int(args.batch_size_end),
        "sampleMatrix": sample_matrix,
        "tag": args.tag,
        "data": args.data,
        "use_dynamic_batch_size": getattr(args, "dynamic_batch_size", False),

        "hierarchical": hierarchical,
        "stage_ratio": stage_ratio,
        "in_degree_max": in_degree_max,
        "max_stages": max_stages,
    }

    print(f"Parameters: radius={radius_val} ({distance_mode}), tangent={tangent_val}, alpha={alpha:.1f}, hierarchical={hierarchical}")
    if hierarchical:
        print(f"  stage_ratio={stage_ratio}, in_degree_max={in_degree_max}, max_stages={max_stages}")

    matrix = VisualMatrix3D(DF, param_dict, "dummy")

    combined_score, mse, spatial_metric = computeV2V4MSE(DF, matrix.matrixW)

    actual_params = {
        "radius": radius_val,
        "tangent": tangent_val,
        "num_degree": eff_num_degree,
        "alpha": alpha,
        "mse": mse,
        "hierarchical": hierarchical,
        "stage_ratio": stage_ratio,
        "in_degree_max": in_degree_max,
        "max_stages": max_stages,
    }

    class Args:
        def __init__(self):
            self.data = args.data
            self.tag = args.tag
            self.algo = args.algo
            self.mode = args.mode

    save_args = Args()

    # Base suffix: distance, angle, alpha
    param_suffix = f"_{radius_val:.2f}_{tangent_val:.2f}_{alpha:.1f}"

    _, pred_colors_array = save_baseline_results(
        DF,
        matrix.matrixW,
        save_args,
        actual_params,
        (combined_score, mse, spatial_metric),
        param_suffix=param_suffix,
        node_generation_order=matrix.node_generation_order,
        batch_info=matrix.batch_info,
    )

    print("\nPerformance Metrics:")
    print(f"  Pure MSE: {mse:.6f}")
    print("Simulation complete")

    return DF, matrix, pred_colors_array


def main():
    parser = argparse.ArgumentParser(description="Visual Cortex Simulation")

    # Data parameters
    parser.add_argument("--data", type=str, default="R1_gpr_grid")
    parser.add_argument("--tag", type=str, default="lh", choices=["lh", "rh"])
    parser.add_argument("--algo", type=str, default="deterministic", choices=["deterministic", "stochastic"])

    # Params
    parser.add_argument("--num_degree", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--tangent", type=float, default=30.0)

    # hierarchical args
    parser.add_argument("--hierarchical", action="store_true", help="Enable hierarchical stage-wise model")
    parser.add_argument("--stage-ratio", dest="stage_ratio", type=float, default=0.90, help="Stop stage when this fraction of sources have outdeg>=1")
    parser.add_argument("--in-degree-max", dest="in_degree_max", type=int, default=1, help="Next source if target in-degree <= this")
    parser.add_argument("--max-stages", dest="max_stages", type=int, default=50, help="Maximum hierarchical stages")

    # Execution parameters
    parser.add_argument("--action", type=str, default="run", choices=["run", "video"])
    parser.add_argument("--sim_mode", type=str, default="fit", choices=["fit", "record"])
    parser.add_argument("--mode", type=str, default="mds", choices=["mds", "euclidean", "sphere"])

    parser.add_argument("--batch_size_start", type=int, default=1)
    parser.add_argument("--batch_size_end", type=int, default=1)
    parser.add_argument("--distance_mode", type=str, default="polar", choices=["polar", "arc", "euclidean"])
    parser.add_argument("--dynamic_batch_size", action="store_true", help="If set, compute batch size schedule dynamically")

    args = parser.parse_args()

    if args.action == "run":
        runSimulation(args)
    elif args.action == "video":
        DF, matrix, pred_colors_array = runSimulation(args)
        create_video_animation(
            args.data,
            args.tag,
            args.mode,
            alpha=float(getattr(args, "alpha", 1.0)),
            euclidean=args.radius,
            tangent=args.tangent,
            DF=DF,
            matrix=matrix,
            pred_colors_array=pred_colors_array,
            distance_mode=args.distance_mode,
        )


if __name__ == "__main__":
    main()
