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

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    print("skopt not available, falling back to dual_annealing")
    SKOPT_AVAILABLE = False

DIAMETER_SCALE_FACTORS = {
    "euclidean": 1.0,
    "mds": 77.667 / 44.8065,
    "sphere": 228.284 / 44.8065,
}

def get_scaled_bounds(mode):
    """Get fixed bounds for all modes."""
    fixed_bounds = [(0.5, 5.0), (5.0, 30.0)]
    return fixed_bounds

def get_scaled_initial_vals(initial_vals, mode):
    """Get fixed initial values for all modes."""
    fixed_vals = [0.5, -1.5, 300.0]
    return fixed_vals


def wrapper(x, data, mode='mds', min_degree=1, max_degree=1, batch_size_start=100, batch_size_end=40, tag="lh"):
    param = {
        "radius": x[0],
        "tangent": x[1],
        "num_degree": 1,
        "alpha": 0.8,
        "mode": "fit",
        "coordinate_mode": mode,
        "min_degree": int(min_degree),
        "max_degree": int(max_degree),
        "batch_size_start": int(batch_size_start),
        "batch_size_end": int(batch_size_end),
        "sampleMatrix": -1,
        "tag": tag
    }
    print(f"  Testing: radius={x[0]:.4f}, tangent={x[1]:.4f}, alpha=0.8000")
    matrix = VisualMatrix3D(data, param, "dummy")
    return matrix.indicator

def parameterSearch(bounds, initialVals, data="X1", tag="lh", mode='mds', n_calls=200, min_degree=1, max_degree=3, batch_size_start=100, batch_size_end=40):
    DF = loadDataDF(data, tag, mode)
    
    mode_dir = f"../results/{mode}"
    os.makedirs(mode_dir, exist_ok=True)
    param_csv = os.path.join(mode_dir, f"params_{data}_{tag}.csv")
    with open(param_csv, "w", newline="\n") as fout:
        fout.write("radius,tangent,alpha,mse\n")
    
    if SKOPT_AVAILABLE:
        space = [
            Real(bounds[0][0], bounds[0][1], name='radius'),
            Real(bounds[1][0], bounds[1][1], name='tangent'),
        ]
        
        print(f"Starting TPE optimization for {data}_{tag} in {mode} mode (n_calls={n_calls})...")
        
        def on_step(res):
            last_x = res.x_iters[-1]
            last_fun = res.func_vals[-1]
            with open(param_csv, "a", newline="\n") as fout:
                fout.write(f"{last_x[0]:.6f},{last_x[1]:.6f},0.800000,{last_fun:.6f}\n")
        
        result = gp_minimize(
            func=lambda params: wrapper(params, DF, mode, min_degree, max_degree, batch_size_start, batch_size_end, tag),
            dimensions=space,
            n_calls=n_calls,  # Number of evaluations
            random_state=42,
            verbose=True,
            callback=[on_step]
        )
        
        best_params = result.x
        best_score = result.fun
        print(f"\nBest parameters found:")
        print(f"  Radius: {best_params[0]:.4f}")
        print(f"  Tangent: {best_params[1]:.4f}")
        print(f"  Alpha: 0.8000")
        print(f"  Best MSE: {best_score:.6f}")

        param_txt = f"../results/{mode}/params_{data}_{tag}.txt"
        os.makedirs(os.path.dirname(param_txt), exist_ok=True)
        with open(param_txt, "w") as f:
            f.write(f"{best_params[0]:.6f}\n{best_params[1]:.6f}\n0.800000\n")
        print(f"Saved optimized parameters to {param_txt}")
        
    else:
        print("Using dual_annealing optimization...")
        opt = dual_annealing(
            lambda params, data_df, coord_mode, dist_mode, min_d, max_d, bs_start, bs_end, tag_param: wrapper(params, data_df, coord_mode, dist_mode, min_d, max_d, bs_start, bs_end, tag_param), 
            bounds, 
            args=(DF, mode, min_degree, max_degree, batch_size_start, batch_size_end, tag), 
            x0=initialVals, 
            maxfun=30, 
            no_local_search=True
        )
        
        best_params = opt.x
        best_score = opt.fun
        result = None
        print(f"\nBest parameters found:")
        print(f"  Radius: {best_params[0]:.4f}")
        print(f"  Tangent: {best_params[1]:.4f}")
        print(f"  Alpha: {best_params[2]:.4f}")
        print(f"  Best MSE: {best_score:.6f}")
        
        param_txt = f"../results/{mode}/params_{data}_{tag}.txt"
        os.makedirs(os.path.dirname(param_txt), exist_ok=True)
        np.savetxt(param_txt, best_params, fmt="%.6f")
        print(f"Saved optimized parameters to {param_txt}")
    
    param = {
        "radius": best_params[0],
        "tangent": best_params[1],
        "alpha": 0.8,
        "mode": "fit",
        "coordinate_mode": mode,
        "min_degree": int(min_degree),
        "max_degree": int(max_degree),
        "batch_size_start": int(batch_size_start),
        "batch_size_end": int(batch_size_end),
        "sampleMatrix": -1,
        "tag": tag,
        "use_dynamic_batch_size": False
    }
    matrix = VisualMatrix3D(DF, param, "dummy")
    mse_only, _, _ = computeV2V4MSE(DF, matrix.matrixW)
    
    with open(param_csv, "a", newline="\n") as fout:
        fout.write(f"{best_params[0]:.6f},{best_params[1]:.6f},0.800000,{mse_only:.6f}\n")
    
    class Args:
        def __init__(self):
            self.data = data
            self.tag = tag
            self.algo = "deterministic"
            self.mode = mode
    
    args = Args()
    actual_params = {
        'radius': best_params[0],
        'tangent': best_params[1],
        'alpha': 0.8,
        'mse': mse_only
    }
    save_baseline_results(DF, matrix.matrixW, args, actual_params)

    return {
        'radius': best_params[0],
        'tangent': best_params[1],
        'alpha': 0.8,
        'result': result if SKOPT_AVAILABLE else None
    }

def gridSearch(data="X1", tag="lh", mode='mds', min_degree=1, max_degree=3, batch_size_start=100, batch_size_end=40):
    """Grid search over parameter combinations"""
    DF = loadDataDF(data, tag, mode)
    
    mode_dir = f"../results/{mode}"
    os.makedirs(mode_dir, exist_ok=True)
    
    gaussian_values = np.array([1, 2, 3, 4, 5])
    fanOut_values = np.array([1, 2, 3, 4, 5])
    axisScale_values = np.arange(50, 301, 50)  # 50, 100, 150, 200, 250, 300
    
    total_combinations = len(gaussian_values) * len(fanOut_values) * len(axisScale_values)
    print(f"Grid search: {total_combinations} parameter combinations")
    print(f"Gaussian: {gaussian_values}")
    print(f"FanOut: {fanOut_values}")
    print(f"AxisScale: {axisScale_values}")
    
    results = []
    
    for i, gaussian in enumerate(gaussian_values):
        for j, fanOut in enumerate(fanOut_values):
            for k, axisScale in enumerate(axisScale_values):
                combo_idx = i * len(fanOut_values) * len(axisScale_values) + j * len(axisScale_values) + k + 1
                print(f"\n=== Combination {combo_idx}/{total_combinations}: gaussian={gaussian:.1f}, fanOut={fanOut:.1f}, axisScale={axisScale:.0f} ===")
                
                param = {
                    "radius": gaussian,
                    "tangent": fanOut,
                    "alpha": axisScale,
                    "mode": "fit",
                    "coordinate_mode": mode,
                    "min_degree": int(min_degree),
                    "max_degree": int(max_degree),
                    "batch_size_start": int(batch_size_start),
                    "batch_size_end": int(batch_size_end),
                    "sampleMatrix": -1,
                    "tag": tag,
                    "use_dynamic_batch_size": False
                }
                
                matrix = VisualMatrix3D(DF, param, "dummy")
                mse_only, _, _ = computeV2V4MSE(DF, matrix.matrixW)
                
                param_suffix = f"_{gaussian:.2f}_{abs(fanOut):.2f}_{axisScale:.2f}"
                
                class Args:
                    def __init__(self):
                        self.data = data
                        self.tag = tag
                        self.algo = "deterministic"
                        self.mode = mode
                
                args = Args()
                actual_params = {
                    'gaussian': gaussian,
                    'fanOut': -fanOut,
                    'axisScale': axisScale,
                    'mse': mse_only
                }
                
                save_baseline_results(DF, matrix.matrixW, args, actual_params, (mse_only, mse_only, None),
                                      param_suffix=param_suffix)
                
                results.append({
                    'gaussian': gaussian,
                    'fanOut': -fanOut,
                    'axisScale': axisScale,
                    'mse': mse_only
                })
                
                print(f"MSE: {mse_only:.6f}")
    
    results_df = pd.DataFrame(results)
    results_file = os.path.join(mode_dir, f"grid_search_{data}_{tag}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nGrid search results saved to: {results_file}")
    
    best_idx = results_df['mse'].idxmin()
    best_result = results_df.iloc[best_idx]
    print(f"\nBest combination:")
    print(f"  Gaussian: {best_result['gaussian']:.1f}")
    print(f"  FanOut: {best_result['fanOut']:.1f}")
    print(f"  AxisScale: {best_result['axisScale']:.0f}")
    print(f"  MSE: {best_result['mse']:.6f}")
    
    return results_df

def load_parameters_from_file(data, tag, mode):
    """Load parameters from txt file like baseline.
    Priority: own data -> R1 -> defaults (larger angle penalty).
    """
    param_txt = f"../results/{mode}/params_{data}_{tag}.txt"
    try:
        loaded_params = np.loadtxt(param_txt)
        print(f"Loaded parameters from {param_txt}: {loaded_params}")
        return loaded_params[0], loaded_params[1], loaded_params[2]
    except FileNotFoundError:
        print(f"No parameter file found at {param_txt}")
        if data != "R1":
            fallback_txt = f"../results/{mode}/params_R1_{tag}.txt"
            try:
                loaded_params = np.loadtxt(fallback_txt)
                print(f"Loaded fallback parameters from {fallback_txt}: {loaded_params}")
                return loaded_params[0], loaded_params[1], loaded_params[2]
            except FileNotFoundError:
                print(f"No fallback parameter file found at {fallback_txt}")
        print("Using default parameters...")
        return 0.5, -1.5, 300.0
    except Exception as e:
        print(f"Error loading parameters: {e}")
        print("Using default parameters...")
        return 0.5, -1.5, 300.0

def runSimulation(args):
    import os
    DF = loadDataDF(args.data, args.tag, args.mode)

    distance_mode = getattr(args, "distance_mode", "polar")
    print(f"Using distance mode: {distance_mode}")
    eff_num_degree = int(getattr(args, "num_degree", 1))
    alpha = float(getattr(args, "alpha", 0.1))
    radius_threshold = float(args.radius)
    tangent_val = args.tangent
    if distance_mode == "polar":
        print(f"Parameters: sigma_r={radius_threshold}, sigma_a={tangent_val}, alpha={alpha:.1f} (num_degree fixed to {eff_num_degree})")
    else:
        print(f"Parameters: radius={radius_threshold}, tangent={tangent_val}, alpha={alpha:.1f} (num_degree fixed to {eff_num_degree})")
    
    print(f"Coordinate mode: {args.mode}")
    print(f"Simulation mode: {args.sim_mode}")
    print(f"Algorithm: {args.algo}")
    
    param_mode = "fit" if args.sim_mode == "fit" else "record"
    sample_matrix = -1 if args.algo == "deterministic" else 1

    param_dict = {
        "mode": param_mode,
        "coordinate_mode": args.mode,
        "num_degree": eff_num_degree,
        "alpha": alpha,
        "radius": radius_threshold,
        "tangent": tangent_val,
        "distance_mode": distance_mode,
        "batch_size_start": int(args.batch_size_start),
        "batch_size_end": int(args.batch_size_end),
        "sampleMatrix": sample_matrix,
        "tag": args.tag,
        "data": args.data,
        "use_dynamic_batch_size": getattr(args, "dynamic_batch_size", False),
        "custom_batch_mode": getattr(args, "custom_batch_mode", None),
    }
    
    matrix = VisualMatrix3D(DF, param_dict, "dummy")
    combined_score, mse, spatial_metric = computeV2V4MSE(DF, matrix.matrixW)

    actual_params = {
        'radius': radius_threshold,
        'tangent': tangent_val,
        'num_degree': eff_num_degree,
        'alpha': alpha,
        'mse': mse
    }
    
    class Args:
        def __init__(self):
            self.data = args.data
            self.tag = args.tag
            self.algo = args.algo
            self.mode = args.mode

    save_args = Args()
    param_suffix = f"_{radius_threshold:.2f}_{tangent_val:.2f}_{alpha:.2f}"
    if getattr(args, "custom_batch_mode", None):
        param_suffix += f"_{args.custom_batch_mode}"
    
    tsv_only = bool(getattr(args, "tsv_only", False))
    ref_colors_path = getattr(args, "ref_colors", None)
    _, pred_colors_array = save_baseline_results(
        DF,
        matrix.matrixW,
        save_args,
        actual_params,
        (combined_score, mse, spatial_metric),
        param_suffix=param_suffix,
        node_generation_order=None if tsv_only else matrix.node_generation_order,
        batch_info=None if tsv_only else matrix.batch_info,
        tsv_only=tsv_only,
        distance_mode=distance_mode,
        ref_colors_path=ref_colors_path,
    )
    
    print(f"\nPerformance Metrics:")
    print(f"  Pure MSE: {mse:.6f}")
    print(f"Simulation complete")
    
    return DF, matrix, pred_colors_array


def main():
    parser = argparse.ArgumentParser(description="Visual Cortex Simulation")
    
    parser.add_argument("--data", type=str, default="R1_gpr_grid")
    parser.add_argument("--tag", type=str, default="lh", choices=["lh", "rh"], help="Hemisphere tag")
    parser.add_argument("--algo", type=str, default="deterministic", choices=["deterministic", "stochastic"], help="Sampling algorithm")
    parser.add_argument("--param_search", type=str, default="predefine", choices=["search", "predefine", "grid"], help="Parameter search mode")
    
    parser.add_argument("--num_degree", type=int, default=1, help="Number of V1 parents per V2-V4 node")
    parser.add_argument("--alpha", type=float, default=0.4, help="Resource decay weight in [0,1]")
    parser.add_argument("--radius", type=float, default=2.0, help="Radial kernel parameter")
    parser.add_argument("--tangent", type=float, default=2.0, help="Tangential kernel parameter (degrees)")
    parser.add_argument(
        "--tsv_only",
        action="store_true",
        help="If set, only saves the predicted tuning TSV and skips W/plots (and any video-related artifacts).",
    )
    parser.add_argument("--action", type=str, default="run", choices=["run", "video"], help="run=simulation, video=create animation")
    parser.add_argument("--sim_mode", type=str, default="fit", choices=["fit", "record"], help="fit=normal simulation, record=intermediate steps")
    parser.add_argument("--mode", type=str, required=True, help="Mode of the simulation")
    parser.add_argument("--distance_mode", type=str, default="polar", choices=["polar", "arc", "euclidean"],
                        help="Distance kernel: polar (rotated elliptical, default), arc (arc+radius), or euclidean (3D)")
    parser.add_argument("--batch_size_start", type=int, default=1, help="Start batch size (default 1)")
    parser.add_argument("--batch_size_end", type=int, default=1, help="End batch size (default 1)")
    parser.add_argument("--min_degree", type=int, default=1, help="Minimum degree for connectivity")
    parser.add_argument("--max_degree", type=int, default=3, help="Maximum degree for connectivity")
    parser.add_argument("--dynamic_batch_size", action="store_true", help="If set, compute batch size schedule dynamically based on V2-V4 gap sweep")
    parser.add_argument("--custom_batch_mode", type=str, default=None,
                        help="Custom spatial batch mode, e.g. custom_angle_up, custom_polar_down, custom_x_random")
    parser.add_argument("--ref_colors", type=str, default=None,
                        help="Path to .npz with pre-computed V1 colors for split (dorsal/ventral) visualization")
    args = parser.parse_args()
    
    if args.action == "run":
        if args.param_search == "search":
            bounds = get_scaled_bounds(args.mode)
            initial_vals = get_scaled_initial_vals([args.radius, args.tangent, args.alpha], args.mode)
            search_result = parameterSearch(bounds, initial_vals, args.data, args.tag, args.mode, 
                                           n_calls=args.n_calls,
                                           min_degree=1, max_degree=1,
                                           batch_size_start=args.batch_size_start, batch_size_end=args.batch_size_end)
            
            print("\n" + "="*60)
            print("Parameter search completed.")
            print("="*60)
        elif args.param_search == "grid":
            gridSearch(args.data, args.tag, args.mode, args.min_degree, args.max_degree,
                      batch_size_start=args.batch_size_start, batch_size_end=args.batch_size_end)
        else:
            runSimulation(args)
    elif args.action == "video":
        DF, matrix, pred_colors_array = runSimulation(args)
        radius_for_video = float(getattr(args, "radius", 6.0))
        create_video_animation(
            args.data,
            args.tag,
            args.mode,
            alpha=float(getattr(args, "alpha", 1.0)),
            euclidean=radius_for_video,
            tangent=args.tangent,
            DF=DF,
            matrix=matrix,
            pred_colors_array=pred_colors_array,
            distance_mode=getattr(args, "distance_mode", "polar"),
            custom_batch_mode=getattr(args, "custom_batch_mode", None),
        )

if __name__ == "__main__":
    main()
