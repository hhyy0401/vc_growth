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

# Global diameter scale factors based on actual measurements
# - Euclidean (3D)  Avg. Diameter: 44.8065 
# - MDS (2D)        Avg. Diameter: 77.667  
# - Sphere          Avg. Diameter: 228.284 
DIAMETER_SCALE_FACTORS = {
    "euclidean": 1.0,
    "mds": 77.667 / 44.8065,  # 1.733
    "sphere": 228.284 / 44.8065  # 5.095
}

def get_scaled_bounds(mode):
    """Search bounds for the two free parameters (sigma_R, sigma_T)."""
    fixed_bounds = [(0.5, 5.0), (0.5, 5.0)]  # radius, tangent
    return fixed_bounds

def get_scaled_initial_vals(initial_vals, mode):
    """Starting point for the fallback optimizer (sigma_R, sigma_T)."""
    return [1.30, 2.20]



def wrapper(x, data, mode='mds', min_degree=1, max_degree=1, batch_size_start=100, batch_size_end=40, tag="lh"):
    param = {
        "radius": x[0],      # radius (polarModel expects this key)
        "tangent": x[1],       # tangent (polarModel expects this key)
        "num_degree": 1,
        "mode": "fit",
        "coordinate_mode": mode,
        "min_degree": int(min_degree),
        "max_degree": int(max_degree),
        "batch_size_start": int(batch_size_start),
        "batch_size_end": int(batch_size_end),
        "sampleMatrix": -1,  # deterministic
        "tag": tag
    }
    print(f"  Testing: radius={x[0]:.4f}, tangent={x[1]:.4f}")
    matrix = VisualMatrix3D(data, param, "dummy")
    return matrix.indicator

def parameterSearch(bounds, initialVals, data="R1_gpr_grid", tag="lh", mode='mds', n_calls=200, min_degree=1, max_degree=3, batch_size_start=100, batch_size_end=40):
    DF = loadDataDF(data, tag, mode)
    
    # Create CSV file for parameter logging
    mode_dir = f"../outputs/predictions/{mode}"
    os.makedirs(mode_dir, exist_ok=True)
    param_csv = os.path.join(mode_dir, f"params_{data}_{tag}.csv")
    with open(param_csv, "w", newline="\n") as fout:
        fout.write("radius,tangent,mse\n")
    
    if SKOPT_AVAILABLE:
        # Use TPE optimization like baseline
        space = [
            Real(bounds[0][0], bounds[0][1], name='radius'),
            Real(bounds[1][0], bounds[1][1], name='tangent'),
        ]
        
        print(f"Starting TPE optimization for {data}_{tag} in {mode} mode (n_calls={n_calls})...")
        
        # Callback to save intermediate results to CSV
        def on_step(res):
            last_x = res.x_iters[-1]
            last_fun = res.func_vals[-1]
            with open(param_csv, "a", newline="\n") as fout:
                fout.write(f"{last_x[0]:.6f},{last_x[1]:.6f},{last_fun:.6f}\n")
        
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
        print(f"  Best MSE: {best_score:.6f}")

        # Save parameters to txt file like baseline
        param_txt = f"../outputs/predictions/{mode}/params_{data}_{tag}.txt"
        os.makedirs(os.path.dirname(param_txt), exist_ok=True)
        with open(param_txt, "w") as f:
            f.write(f"{best_params[0]:.6f}\n{best_params[1]:.6f}\n")
        print(f"Saved optimized parameters to {param_txt}")
        
    else:
        # Fallback to dual_annealing
        print("Using dual_annealing optimization...")
        opt = dual_annealing(
            wrapper,
            bounds,
            args=(DF, mode, min_degree, max_degree, batch_size_start, batch_size_end, tag),
            x0=initialVals,
            maxfun=30,
            no_local_search=True
        )

        best_params = opt.x
        best_score = opt.fun
        result = None  # dual_annealing doesn't return result object
        print(f"\nBest parameters found:")
        print(f"  Radius: {best_params[0]:.4f}")
        print(f"  Tangent: {best_params[1]:.4f}")
        print(f"  Best MSE: {best_score:.6f}")
        
        # Save parameters to txt file
        param_txt = f"../outputs/predictions/{mode}/params_{data}_{tag}.txt"
        os.makedirs(os.path.dirname(param_txt), exist_ok=True)
        np.savetxt(param_txt, best_params, fmt="%.6f")
        print(f"Saved optimized parameters to {param_txt}")
    
    # Run with best parameters and save all outputs
    param = {
        "radius": best_params[0],  # radius
        "tangent": best_params[1],       # tangent
        "mode": "fit",
        "coordinate_mode": mode,
        "min_degree": int(min_degree),
        "max_degree": int(max_degree),
        "batch_size_start": int(batch_size_start),
        "batch_size_end": int(batch_size_end),
        "sampleMatrix": -1,  # deterministic
        "tag": tag,
        "use_dynamic_batch_size": False
    }
    matrix = VisualMatrix3D(DF, param, "dummy")
    mse_only, _, _ = computeV2V4MSE(DF, matrix.matrixW)
    
    with open(param_csv, "a", newline="\n") as fout:
        fout.write(f"{best_params[0]:.6f},{best_params[1]:.6f},{mse_only:.6f}\n")
    
    # Save best results in baseline format (no plot generation)
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
        'mse': mse_only
    }
    save_baseline_results(DF, matrix.matrixW, args, actual_params)

    # Return best parameters for video generation
    return {
        'radius': best_params[0],
        'tangent': best_params[1],
        'result': result if SKOPT_AVAILABLE else None
    }

def gridSearch(data="R1_gpr_grid", tag="lh", mode='mds', min_degree=1, max_degree=3, batch_size_start=100, batch_size_end=40):
    """Grid search over parameter combinations"""
    DF = loadDataDF(data, tag, mode)
    
    # Create output directory
    mode_dir = f"../outputs/predictions/{mode}"
    os.makedirs(mode_dir, exist_ok=True)
    
    # Grid parameters (sigma_R, sigma_T)
    radius_values = np.arange(0.5, 3.01, 0.5)
    tangent_values = np.arange(0.5, 3.01, 0.5)

    total_combinations = len(radius_values) * len(tangent_values)
    print(f"Grid search: {total_combinations} parameter combinations")
    print(f"Radius (sigma_R): {radius_values}")
    print(f"Tangent (sigma_T): {tangent_values}")

    results = []

    for i, radius in enumerate(radius_values):
        for j, tangent in enumerate(tangent_values):
            combo_idx = i * len(tangent_values) + j + 1
            print(f"\n=== Combination {combo_idx}/{total_combinations}: radius={radius:.2f}, tangent={tangent:.2f} ===")

            # Create parameter dict with grid values
            param = {
                "radius": radius,
                "tangent": tangent,
                "mode": "fit",
                "coordinate_mode": mode,
                "min_degree": int(min_degree),
                "max_degree": int(max_degree),
                "batch_size_start": int(batch_size_start),
                "batch_size_end": int(batch_size_end),
                "sampleMatrix": -1,  # deterministic
                "tag": tag,
                "use_dynamic_batch_size": False
            }

            # Run simulation
            matrix = VisualMatrix3D(DF, param, "dummy")
            mse_only, _, _ = computeV2V4MSE(DF, matrix.matrixW)

            # Create filename suffix with parameter values (rounded to 2 decimals)
            param_suffix = f"_{radius:.2f}_{tangent:.2f}"

            # Save results with parameter-specific filenames
            class Args:
                def __init__(self):
                    self.data = data
                    self.tag = tag
                    self.algo = "deterministic"
                    self.mode = mode

            args = Args()
            actual_params = {
                'radius': radius,
                'tangent': tangent,
                'mse': mse_only
            }

            # Save with parameter-specific filenames
            save_baseline_results(DF, matrix.matrixW, args, actual_params, (mse_only, mse_only, None),
                                  param_suffix=param_suffix)

            results.append(actual_params)

            print(f"MSE: {mse_only:.6f}")

    # Save grid search results summary
    results_df = pd.DataFrame(results)
    results_file = os.path.join(mode_dir, f"grid_search_{data}_{tag}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nGrid search results saved to: {results_file}")

    # Find best combination
    best_idx = results_df['mse'].idxmin()
    best_result = results_df.iloc[best_idx]
    print(f"\nBest combination:")
    print(f"  Radius: {best_result['radius']:.2f}")
    print(f"  Tangent: {best_result['tangent']:.2f}")
    print(f"  MSE: {best_result['mse']:.6f}")

    return results_df

def load_parameters_from_file(data, tag, mode):
    """Load (sigma_R, sigma_T) from a params txt file.
    Priority: own data -> R1_gpr_grid -> published defaults.
    """
    param_txt = f"../outputs/predictions/{mode}/params_{data}_{tag}.txt"
    try:
        loaded_params = np.loadtxt(param_txt)
        print(f"Loaded parameters from {param_txt}: {loaded_params}")
        return loaded_params[0], loaded_params[1]
    except FileNotFoundError:
        print(f"No parameter file found at {param_txt}")
        # Try fallback to the NMT template subject
        if data != "R1_gpr_grid":
            fallback_txt = f"../outputs/predictions/{mode}/params_R1_gpr_grid_{tag}.txt"
            try:
                loaded_params = np.loadtxt(fallback_txt)
                print(f"Loaded fallback parameters from {fallback_txt}: {loaded_params}")
                return loaded_params[0], loaded_params[1]
            except FileNotFoundError:
                print(f"No fallback parameter file found at {fallback_txt}")
        print("Using published default parameters...")
        return 1.30, 2.20
    except Exception as e:
        print(f"Error loading parameters: {e}")
        print("Using published default parameters...")
        return 1.30, 2.20

def runSimulation(args):
    import os
    DF = loadDataDF(args.data, args.tag, args.mode)

    distance_mode = getattr(args, "distance_mode", "polar")
    print(f"Using distance mode: {distance_mode}")
    # num_degree defaults to 1; can be overridden via --num_degree
    eff_num_degree = int(getattr(args, "num_degree", 1))
    # Radius threshold replaces legacy euclidean threshold (no more --euclidean)
    radius_threshold = float(args.radius)
    tangent = args.tangent
    if distance_mode == "polar":
        print(f"Parameters: sigma_r={radius_threshold}, sigma_a={tangent} (num_degree fixed to {eff_num_degree})")
    else:
        print(f"Parameters: radius={radius_threshold}, tangent={tangent} (num_degree fixed to {eff_num_degree})")

    print(f"Coordinate mode: {args.mode}")
    print(f"Simulation mode: {args.sim_mode}")
    print(f"Algorithm: {args.algo}")
    
    # Map sim_mode to param mode
    param_mode = "fit" if args.sim_mode == "fit" else "record"
    
    # Map algo to sampleMatrix
    sample_matrix = -1 if args.algo == "deterministic" else 1
    
    # Build parameter dict
    param_dict = {
        "mode": param_mode,
        "coordinate_mode": args.mode,
        "num_degree": eff_num_degree,
        "radius": radius_threshold,
        "tangent": tangent,
        "distance_mode": distance_mode,
        "batch_size_start": int(args.batch_size_start),
        "batch_size_end": int(args.batch_size_end),
        "sampleMatrix": sample_matrix,
        "tag": args.tag,
        "data": args.data,
        "use_dynamic_batch_size": getattr(args, "dynamic_batch_size", False),
        "custom_batch_mode": getattr(args, "custom_batch_mode", None),
    }
    
    # Run simulation
    matrix = VisualMatrix3D(DF, param_dict, "dummy")
    
    # Compute performance metric (MSE-only)
    combined_score, mse, spatial_metric = computeV2V4MSE(DF, matrix.matrixW)
    
    # Save results with actual parameters used
    actual_params = {
        'radius': radius_threshold,
        'tangent': tangent,
        'num_degree': eff_num_degree,
        'mse': mse
    }
    
    # Create Args object for save_baseline_results
    class Args:
        def __init__(self):
            self.data = args.data
            self.tag = args.tag
            self.algo = args.algo
            self.mode = args.mode
    
    save_args = Args()
    # Create param_suffix for filename
    param_suffix = f"_{radius_threshold:.2f}_{tangent:.2f}"
    if getattr(args, "custom_batch_mode", None):
        param_suffix += f"_{args.custom_batch_mode}"  # e.g. _polar_fp, _polar_pf, _polar_random
    
    # Save node_generation_order and batch_info with W file for video generation
    # Always generates plot and returns pred_colors_array
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
    
    # Data parameters
    parser.add_argument("--data", type=str, default="R1_gpr_grid",
                        help="Data identifier (R1_gpr_grid, S1_gpr_grid, ..., S6_gpr_grid)")
    parser.add_argument("--tag", type=str, default="lh", choices=["lh", "rh"], help="Hemisphere tag")
    parser.add_argument("--algo", type=str, default="deterministic", choices=["deterministic", "stochastic"], help="Sampling algorithm")
    parser.add_argument("--param_search", type=str, default="predefine", choices=["search", "predefine", "grid"], help="Parameter search mode")
    parser.add_argument("--n_calls", type=int, default=200, help="Number of evaluations for --param_search search")
    
    # Parameters
    # num_degree is fixed to 1 in this model; the argument is accepted but not varied.
    parser.add_argument("--num_degree", type=int, default=1, help="Connection degree (fixed to 1).")
    parser.add_argument("--radius", type=float, default=1.30, help="Radial kernel width sigma_R.")
    parser.add_argument("--tangent", type=float, default=2.20, help="Tangential kernel width sigma_T.")
    # Output control
    parser.add_argument(
        "--tsv_only",
        action="store_true",
        help="If set, only saves the predicted tuning TSV and skips W/plots (and any video-related artifacts).",
    )
    # Execution parameters
    parser.add_argument("--action", type=str, default="run", choices=["run", "video"], help="run=simulation, video=create animation")
    parser.add_argument("--sim_mode", type=str, default="fit", choices=["fit", "record"], help="fit=normal simulation, record=intermediate steps")
    parser.add_argument("--mode", type=str, required=True, help="Mode of the simulation")
    parser.add_argument("--distance_mode", type=str, default="polar", choices=["polar", "arc", "euclidean", "sphere_geo"],
                        help="Distance kernel: polar (rotated elliptical, default), arc (arc+radius), euclidean (3D), or sphere_geo (great-circle on radius-100 sphere)")
    parser.add_argument("--batch_size_start", type=int, default=1, help="Start batch size (default 1)")
    parser.add_argument("--batch_size_end", type=int, default=1, help="End batch size (default 1)")
    parser.add_argument("--min_degree", type=int, default=1, help="Minimum degree for connectivity")
    parser.add_argument("--max_degree", type=int, default=3, help="Maximum degree for connectivity")
    parser.add_argument("--dynamic_batch_size", action="store_true", help="If set, compute batch size schedule dynamically based on V2-V4 gap sweep")
    parser.add_argument("--custom_batch_mode", type=str, default=None,
                        help="Custom spatial batch mode, {angle|polar|euclidean|x}_{fp|pf|random}, e.g. polar_fp")
    parser.add_argument("--ref_colors", type=str, default=None,
                        help="Path to .npz with pre-computed V1 colors for split (dorsal/ventral) visualization")
    args = parser.parse_args()
    
    if args.action == "run":
        if args.param_search == "search":
            # Parameter search bounds
            bounds = get_scaled_bounds(args.mode)
            initial_vals = get_scaled_initial_vals([args.radius, args.tangent], args.mode)
            # Use constant degree=1; batch sizes from args
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
        # Use runSimulation to run simulation once and save results
        DF, matrix, pred_colors_array = runSimulation(args)
        
        # Create video animation using saved results (no re-simulation)
        # For backward-compatible filenames, pass the radius value as the legacy 'euclidean' argument.
        radius_for_video = float(getattr(args, "radius", 1.30))
        create_video_animation(
            args.data,
            args.tag,
            args.mode,
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
