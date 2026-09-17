import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy version.*")

import argparse

from polarModel import VisualMatrix3D
from utils import loadDataDF, computeV2V4MSE, save_baseline_results


def runSimulation(args):
    """Grow the model on one hemisphere and save the predictions."""
    DF = loadDataDF(args.data, args.tag)
    radius = args.radius if args.radius is not None else (1.40 if args.hierarchical else 1.30)
    print(f"Parameters: sigma_R={radius}, sigma_T={args.tangent}, degree={args.num_degree}"
          f"{', hierarchical' if args.hierarchical else ''}")

    param = {
        "radius": float(radius),
        "tangent": float(args.tangent),
        "num_degree": int(args.num_degree),
        "tag": args.tag,
        "data": args.data,
        "custom_batch_mode": args.custom_batch_mode,
        "kernel": args.kernel,
    }
    if args.hierarchical:
        from hierarchical import HierarchicalMatrix
        matrix = HierarchicalMatrix(DF, param)
    else:
        matrix = VisualMatrix3D(DF, param)

    mse = computeV2V4MSE(DF, matrix.matrixW)
    param_suffix = f"_{float(radius):.2f}_{float(args.tangent):.2f}"
    if args.custom_batch_mode:
        param_suffix += f"_{args.custom_batch_mode}"
    if args.hierarchical:
        param_suffix += "_hierarchical"
    if args.kernel != "polar":
        param_suffix += f"_{args.kernel}"

    pred_colors_array = save_baseline_results(
        DF,
        matrix.matrixW,
        args,
        mse,
        param_suffix=param_suffix,
        node_generation_order=matrix.node_generation_order,
        batch_info=matrix.batch_info,
        plot=not args.no_plot,
    )
    print(f"\nMSE of V2-V4 tuning: {mse:.6f}")
    return DF, matrix, pred_colors_array


def main():
    parser = argparse.ArgumentParser(description="Growth model of extrastriate retinotopy")
    parser.add_argument("--data", type=str, default="NMT_gpr_grid",
                        help="Dataset (NMT_gpr_grid, M1_gpr_grid, ..., M6_gpr_grid)")
    parser.add_argument("--tag", type=str, default="lh", choices=["lh", "rh"], help="Hemisphere")
    parser.add_argument("--radius", type=float, default=None,
                        help="Radial kernel width sigma_R (default 1.30, or 1.40 with --hierarchical)")
    parser.add_argument("--tangent", type=float, default=2.20, help="Tangential kernel width sigma_T")
    parser.add_argument("--num_degree", type=int, default=1, help="V1 parents per extrastriate node")
    parser.add_argument("--custom_batch_mode", type=str, default=None,
                        help="Spatial growth order, {angle|polar|euclidean|x}_{fp|pf|random}")
    parser.add_argument("--kernel", choices=["polar", "sphere"], default="polar",
                        help="Distance kernel: polar (default), or sphere (geodesics on the "
                             "spherical surface, Supp. Fig. S6)")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Grow in sequential stages instead of from V1 only (Supp. Fig. S7)")
    parser.add_argument("--no_plot", action="store_true", help="Skip the comparison plot")
    parser.add_argument("--video", action="store_true", help="Also write the growth animation as HTML")
    args = parser.parse_args()

    DF, matrix, pred_colors_array = runSimulation(args)

    if args.video:
        from visualizationUtil import create_video_animation
        create_video_animation(
            args.data,
            args.tag,
            radius=float(args.radius if args.radius is not None else 1.30),
            tangent=float(args.tangent),
            DF=DF,
            matrix=matrix,
            pred_colors_array=pred_colors_array,
            custom_batch_mode=args.custom_batch_mode,
        )


if __name__ == "__main__":
    main()
