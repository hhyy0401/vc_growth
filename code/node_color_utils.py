import numpy as np
import pandas as pd
import pickle
import os
import csv
from matplotlib.colors import LinearSegmentedColormap

def getColorMap():
    colors = ["green", "aqua", "pink", "red"]
    positions = np.linspace(0, 1, len(colors))
    custom_map = LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)))
    return custom_map

def restore_tuning_range(tuning_vec):
    if tuning_vec is None:
        return None
    
    if not isinstance(tuning_vec, np.ndarray) or len(tuning_vec) < 2:
        return None
    
    if np.any(np.isnan(tuning_vec[:2])) or np.any(np.isinf(tuning_vec[:2])):
        return None
    
    x_min, x_max = 0.411397, 7.721101
    y_min, y_max = -6.013087, 7.707419
    
    restored_vec = tuning_vec.copy()
    restored_vec[0] = tuning_vec[0] * (x_max - x_min) + x_min
    restored_vec[1] = tuning_vec[1] * (y_max - y_min) + y_min
    
    return restored_vec

def scaleColor(tuning_vectors, absolute=True, bin=True):
    """Scale tuning vectors to color values via arctan2."""
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
        return np.full(len(tuning_vectors), 0.5)
    
    tuning_scalars = np.array(tuning_scalars)
    
    if absolute:
        adjTuning = np.abs(tuning_scalars)
    else:
        adjTuning = tuning_scalars
    
    c_max = np.max(adjTuning)
    c_min = np.min(adjTuning)
    c_range = c_max - c_min
    
    colors = np.full(len(tuning_vectors), 0.5)
    
    if bin:
        if c_range < 1e-9:
            binned_colors = np.full_like(adjTuning, 0.33)
        else:
            normalized = (adjTuning - c_min) / c_range
            binned_colors = np.full_like(normalized, 1.0)  # Start with highest bin (1.0)
            binned_colors[normalized <= 0.75] = 0.67
            binned_colors[normalized <= 0.5] = 0.33
            binned_colors[normalized <= 0.25] = 0.0

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
    """Calculate node colors: V1 binning from true tuning, V2-V4 from weight matrix or predictions."""
    with open(data_file, 'rb') as f:
        raw_data = pickle.load(f)
    
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
    
    V1_node_ids = list(V1_nodes.keys())
    V1_tuning_vectors = [V1_nodes[nid] for nid in V1_node_ids]
    V1_colors = scaleColor(V1_tuning_vectors, absolute=True, bin=True)
    
    result_colors = {}
    for i, node_id in enumerate(V1_node_ids):
        if i < len(V1_colors):
            result_colors[node_id] = V1_colors[i]
    
    if weight_matrix is not None:
        if hasattr(weight_matrix, 'cpu'):
            W = weight_matrix.cpu().numpy()
        else:
            W = weight_matrix
        
        V1_color_array = np.array([result_colors.get(nid, 0.5) for nid in V1_node_ids])

        area_node_ids = list(V2_nodes.keys()) + list(V3_nodes.keys()) + list(V4_nodes.keys())
        
        if len(area_node_ids) > 0 and W.shape[1] > W.shape[0]:
            W_V2_to_V4 = W[:, W.shape[0]:]
            col_sums = np.sum(W_V2_to_V4, axis=0)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            W_V2_to_V4_normalized = W_V2_to_V4 / col_sums[np.newaxis, :]
            VnColors = (W_V2_to_V4_normalized.T @ V1_color_array).flatten()

            for i, node_id in enumerate(area_node_ids):
                if i < len(VnColors):
                    result_colors[node_id] = VnColors[i]
    else:
        area_node_ids = list(V2_nodes.keys()) + list(V3_nodes.keys()) + list(V4_nodes.keys())
        area_tuning_vectors = []
        
        for node_id in area_node_ids:
            if node_id in predicted_tuning_map:
                area_tuning_vectors.append(predicted_tuning_map[node_id])
            else:
                if node_id in V2_nodes:
                    area_tuning_vectors.append(V2_nodes[node_id])
                elif node_id in V3_nodes:
                    area_tuning_vectors.append(V3_nodes[node_id])
                elif node_id in V4_nodes:
                    area_tuning_vectors.append(V4_nodes[node_id])
                else:
                    area_tuning_vectors.append(None)
        
        area_colors = scaleColor(area_tuning_vectors, absolute=True, bin=False)
        
        for i, node_id in enumerate(area_node_ids):
            if i < len(area_colors):
                result_colors[node_id] = area_colors[i]
    
    return result_colors

def save_node_colors_csv(node_colors, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Node_ID", "Color_Value"])
        
        for node_id, color_value in node_colors.items():
            writer.writerow([node_id, f"{color_value:.6f}"])
    
    print(f"Node colors saved to: {output_path}") 