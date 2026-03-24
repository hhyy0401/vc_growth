import numpy as np
import pandas as pd
import pickle
import os
import csv
from matplotlib.colors import LinearSegmentedColormap

def getColorMap():
    """Create a custom LinearSegmentedColormap."""
    colors = ["green", "aqua", "pink", "red"]
    positions = np.linspace(0, 1, len(colors))
    custom_map = LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)))
    return custom_map

def restore_tuning_range(tuning_vec):
    """Restore tuning values from [0,1] range to original range"""
    if tuning_vec is None:
        return None
    
    if not isinstance(tuning_vec, np.ndarray) or len(tuning_vec) < 2:
        return None
    
    if np.any(np.isnan(tuning_vec[:2])) or np.any(np.isinf(tuning_vec[:2])):
        return None
    
    # Original ranges
    x_min, x_max = 0.411397, 7.721101
    y_min, y_max = -6.013087, 7.707419
    
    # Restore from [0,1] to original range
    restored_vec = tuning_vec.copy()
    restored_vec[0] = tuning_vec[0] * (x_max - x_min) + x_min
    restored_vec[1] = tuning_vec[1] * (y_max - y_min) + y_min
    
    return restored_vec

def scaleColor(tuning_vectors, absolute=True, bin=True):
    """Convert 2D tuning vectors to scalar color values with optional binning."""
    tuning_scalars = []
    valid_indices = []
    
    for i, v in enumerate(tuning_vectors):
        if (v is not None and isinstance(v, np.ndarray) 
            and v.ndim > 0 and len(v) >= 2
            and not np.any(np.isnan(v[:2])) and not np.any(np.isinf(v[:2]))):
            tuning_scalar = np.arctan2(v[1], v[0])
            tuning_scalars.append(tuning_scalar)
            valid_indices.append(i)
    
    if not valid_indices:
        return np.full(len(tuning_vectors), 0.5)  # Default gray color
    
    tuning_scalars = np.array(tuning_scalars)
    
    if absolute:
        adjTuning = np.abs(tuning_scalars)
    else:
        adjTuning = tuning_scalars
    
    c_max = np.max(adjTuning)
    c_min = np.min(adjTuning)
    c_range = c_max - c_min
    
    # Initialize output array
    colors = np.full(len(tuning_vectors), 0.5)  # Default gray
    
    if bin:
        if c_range < 1e-9:
            # If all values are the same, assign middle bin
            binned_colors = np.full_like(adjTuning, 0.33)
        else:
            normalized = (adjTuning - c_min) / c_range
            binned_colors = np.full_like(normalized, 1.0)  # Start with highest bin (1.0)
            binned_colors[normalized <= 0.75] = 0.67
            binned_colors[normalized <= 0.5] = 0.33
            binned_colors[normalized <= 0.25] = 0.0
        
        # Assign binned colors to valid indices
        for i, idx in enumerate(valid_indices):
            colors[idx] = binned_colors[i]
    else:
        if c_range < 1e-9:
            normalized = np.zeros_like(adjTuning)
        else:
            normalized = (adjTuning - c_min) / c_range
        
        for i, idx in enumerate(valid_indices):
            colors[idx] = normalized[i]
    
    return colors

def calculate_node_colors_newcode_style(data_file, predicted_tuning_map, weight_matrix=None, mode="baseline"):
    """
    Calculate node colors using newCode methodology:
    1. V1 binning (from true tuning)
    2. V2-V4 weighted colors (from weight matrix if available, otherwise from predicted tuning)
    
    Args:
        data_file: Path to the .pkl data file
        predicted_tuning_map: Dict mapping node_id -> predicted tuning vector
        weight_matrix: Optional weight matrix for V1->V2-V4 propagation
        mode: "baseline" or "evaluate"
    
    Returns:
        Dict mapping node_id -> color value
    """
    # Load raw data
    with open(data_file, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Separate nodes by area
    V1_nodes = {}
    V2_nodes = {}
    V3_nodes = {}
    V4_nodes = {}
    
    for node_id, entry in raw_data.items():
        node_id_str = str(node_id)
        area = entry.get('area')
        tuning = entry.get('tuning')
        
        if area == 1:
            V1_nodes[node_id_str] = restore_tuning_range(np.array(tuning, dtype=float))
        elif area == 2:
            V2_nodes[node_id_str] = restore_tuning_range(np.array(tuning, dtype=float))
        elif area == 3:
            V3_nodes[node_id_str] = restore_tuning_range(np.array(tuning, dtype=float))
        elif area == 4:
            V4_nodes[node_id_str] = restore_tuning_range(np.array(tuning, dtype=float))
    
    # Step 1: V1 binning (always use true tuning for V1)
    V1_node_ids = list(V1_nodes.keys())
    V1_tuning_vectors = [V1_nodes[nid] for nid in V1_node_ids]
    V1_colors = scaleColor(V1_tuning_vectors, absolute=True, bin=True)
    
    # Store V1 colors
    result_colors = {}
    for i, node_id in enumerate(V1_node_ids):
        if i < len(V1_colors):
            result_colors[node_id] = V1_colors[i]
    
    # Step 2: V2-V4 colors
    if weight_matrix is not None:
        # Convert weight matrix to numpy if it's a PyTorch tensor
        if hasattr(weight_matrix, 'cpu'):
            W = weight_matrix.cpu().numpy()
        else:
            W = weight_matrix
        
        # W[:, W.shape[0]:].T @ V1Color
        V1_color_array = np.array([result_colors.get(nid, 0.5) for nid in V1_node_ids])
        
        # Get V2-V4 node order from areas 2, 3, 4
        area_node_ids = list(V2_nodes.keys()) + list(V3_nodes.keys()) + list(V4_nodes.keys())
        
        if len(area_node_ids) > 0 and W.shape[1] > W.shape[0]:
            W_V2_to_V4 = W[:, W.shape[0]:]  # V1 -> V2-V4 weights
            
            # Column-normalize weight matrix so each V2-V4 node receives weights summing to 1
            # This ensures color values stay in 0-1 range
            col_sums = np.sum(W_V2_to_V4, axis=0)  # Sum of weights each V2-V4 node receives
            col_sums = np.where(col_sums == 0, 1, col_sums)  # Avoid division by zero
            W_V2_to_V4_normalized = W_V2_to_V4 / col_sums[np.newaxis, :]  # Normalize each column (V2-V4 node)
            
            # V2-V4 colors = normalized weighted sum of V1 colors 
            VnColors = (W_V2_to_V4_normalized.T @ V1_color_array).flatten()
            
            # Assign weighted colors to V2-V4 nodes
            for i, node_id in enumerate(area_node_ids):
                if i < len(VnColors):
                    result_colors[node_id] = VnColors[i]
    else:
        # Use predicted tuning with continuous coloring (for smooth baseline results)
        area_node_ids = list(V2_nodes.keys()) + list(V3_nodes.keys()) + list(V4_nodes.keys())
        area_tuning_vectors = []
        
        for node_id in area_node_ids:
            # Use predicted tuning if available, otherwise use true tuning
            if node_id in predicted_tuning_map:
                area_tuning_vectors.append(predicted_tuning_map[node_id])
            else:
                # Fall back to true tuning
                if node_id in V2_nodes:
                    area_tuning_vectors.append(V2_nodes[node_id])
                elif node_id in V3_nodes:
                    area_tuning_vectors.append(V3_nodes[node_id])
                elif node_id in V4_nodes:
                    area_tuning_vectors.append(V4_nodes[node_id])
                else:
                    area_tuning_vectors.append(None)
        
        # Use continuous coloring for smooth baseline results
        area_colors = scaleColor(area_tuning_vectors, absolute=True, bin=False)
        
        for i, node_id in enumerate(area_node_ids):
            if i < len(area_colors):
                result_colors[node_id] = area_colors[i]
    
    return result_colors

def save_node_colors_csv(node_colors, output_path):
    """Save node colors to CSV file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Node_ID", "Color_Value"])
        
        for node_id, color_value in node_colors.items():
            writer.writerow([node_id, f"{color_value:.6f}"])
    
    print(f"Node colors saved to: {output_path}") 