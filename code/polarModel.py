from tqdm import tqdm
import torch
import numpy as np
from utils import initDirectory
import sys
sys.path.insert(0, '..')
from TUNING_COLOR_UTILS import compute_tuning_colors

class VisualMatrix3D(object):
    def __init__(self, dataDF, param, outputDir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.outputDir = initDirectory(param, outputDir)
        self.num_degree = int(param.get("num_degree", 2))
        self.alpha = float(param.get("alpha", 0.4))
        self.mode = param.get("coordinate_mode", "sphere")
        self.batch_size_start = int(param.get("batch_size_start", int(param.get("batch_size", 1))))
        self.batch_size_end = int(param.get("batch_size_end", int(param.get("batch_size", 1))))

        self.use_dynamic_batch_size = bool(param.get("use_dynamic_batch_size", False))
        self.batch_size_schedule = []  # Initialize to avoid AttributeError

        self.custom_batch_mode = param.get("custom_batch_mode", None)
        self.custom_node_order = None
        self.custom_batch_assignments = None

        if self.custom_batch_mode:
            from custom_batch import parse_custom_mode, get_custom_node_order
            mode_name, ecc_order = parse_custom_mode(self.custom_batch_mode)
            print(f"Custom batch mode: {self.custom_batch_mode} (mode={mode_name}, ecc={ecc_order})")
            self.custom_node_order, self.custom_batch_assignments = get_custom_node_order(
                dataDF, mode_name, ecc_order, n_batches=30,
                radius=float(param.get("radius", 6.0)),
                tangent_deg=float(param.get("tangent", 30.0)),
            )
            self.batch_size_start = 1
            self.batch_size_end = 1
            self.batch_size = 1
        elif self.use_dynamic_batch_size:
            from utils import compute_dynamic_batch_sizes
            print("Calculating dynamic batch size schedule (on-the-fly)...")
            bs_out_dir = "../batch_size" 
            tag_val = param.get("tag", "untagged")
            data_val = param.get("data", "dynamic_run")
            
            try:
                self.batch_size_schedule = compute_dynamic_batch_sizes(dataDF, output_dir=bs_out_dir, data_name=data_val, tag_name=tag_val)
                print(f"Dynamic schedule generated with {len(self.batch_size_schedule)} steps.")
            except Exception as e:
                print(f"Error calculating dynamic batch sizes: {e}")
                print("Falling back to batch_size = 1.")
                self.batch_size_start = 1
                self.batch_size_end = 1
                self.batch_size = 1
        else:
            self.batch_size_start = 1
            self.batch_size_end = 1
            self.batch_size = 1

        self.radius = float(param.get("radius", 6.0))
        self.tangent = float(param.get("tangent", 30.0))
        self.distance_mode = param.get("distance_mode", "polar")
        self.direct_distance_weight = 1.0

        self.matrixC, self.matrixW, self.matrixD, self.mask = self.initMatrix(
            dataDF,
            radius=self.radius,
            tangent=self.tangent,
            mode=self.mode,
            distance_mode=self.distance_mode,
        )
        _V1 = self.matrixC.shape[0]
        self._cached_propagation = (self.matrixC @ self.matrixD[:_V1, :]).clone()
        self._cached_deg = torch.zeros(_V1, device=self.device, dtype=torch.float32)
        self.dataDF = dataDF
        self.tag = param.get("tag", None)

        V1_tuning = dataDF[dataDF["area"] == 1][["tuningX", "tuningY"]].values
        self.V1_tuning_tensor = torch.tensor(V1_tuning, device=self.device, dtype=torch.float32)
        V1Count = self.matrixC.shape[0]
        self.color_mask = torch.zeros(V1Count, device=self.device, dtype=torch.float32)
        self.record = self.initRecord()
        self.node_generation_order = []
        self.batch_info = []
        self.indicator = self.simulate(dataDF, param)

    def initMatrix(self, DF, radius=6.0, tangent=30.0, mode="sphere", distance_mode="polar"):
        V1Count = DF[(DF["area"] == 1)].shape[0]
        VnCount = DF[(DF["area"] != 1)].shape[0]

        mask = torch.eye(VnCount, device=self.device, dtype=torch.float32)

        x_coords = torch.tensor(DF["x"].values, device=self.device, dtype=torch.float32)
        y_coords = torch.tensor(DF["y"].values, device=self.device, dtype=torch.float32)
        z_coords = torch.tensor(DF["z"].values, device=self.device, dtype=torch.float32)

        x1 = x_coords.unsqueeze(1)
        x2 = x_coords.unsqueeze(0)
        y1 = y_coords.unsqueeze(1)
        y2 = y_coords.unsqueeze(0)
        z1 = z_coords.unsqueeze(1)
        z2 = z_coords.unsqueeze(0)
        euclidean_distance = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

        if distance_mode == "euclidean":
            kernel = torch.exp(-euclidean_distance / radius)
        elif distance_mode == "arc":
            r_coords = torch.tensor(DF["r"].values, device=self.device, dtype=torch.float32)
            t_coords = torch.tensor(DF["t"].values, device=self.device, dtype=torch.float32)
            t1 = t_coords.unsqueeze(1)
            t2 = t_coords.unsqueeze(0)
            theta_diff = t1 - t2
            angle_distance = torch.minimum(torch.abs(theta_diff),
                                        torch.abs(torch.abs(theta_diff) - 2*np.pi))
            r1 = r_coords.unsqueeze(1)
            r2 = r_coords.unsqueeze(0)
            radius_distance = torch.abs(r1 - r2)
            kernel_euclidean = torch.exp(-radius_distance / radius)
            r_min = torch.minimum(r1, r2)
            arc_distance = r_min * angle_distance
            tangent_rad = tangent * np.pi / 180.0
            kernel_tangent = torch.exp(-arc_distance / tangent_rad)
            kernel = kernel_euclidean * kernel_tangent
        else:
            r_coords = torch.tensor(DF["r"].values, device=self.device, dtype=torch.float32)
            t_coords = torch.tensor(DF["t"].values, device=self.device, dtype=torch.float32)

            vx = r_coords * torch.cos(t_coords)
            vy = r_coords * torch.sin(t_coords)

            dx = vx.unsqueeze(0) - vx.unsqueeze(1)
            dy = vy.unsqueeze(0) - vy.unsqueeze(1)

            cos_t = torch.cos(t_coords).unsqueeze(1)
            sin_t = torch.sin(t_coords).unsqueeze(1)

            d_r = dx * cos_t + dy * sin_t
            d_t = -dx * sin_t + dy * cos_t


            d = torch.sqrt((d_r / radius)**2 + (d_t / tangent)**2)
            kernel = torch.exp(-d)

        matrixC = kernel[:V1Count, :V1Count]
        matrixD = kernel[:, V1Count:]
        matrixW = torch.cat([torch.eye(V1Count, device=self.device), torch.zeros(V1Count, VnCount, device=self.device)], dim=1)
        euclidean_v1_vn = euclidean_distance[:V1Count, V1Count:]
        self.shortest_angle_distance_per_vn = torch.min(euclidean_v1_vn, dim=0)[0]
        return matrixC, matrixW, matrixD, mask

    def initRecord(self):
        V1Count = self.matrixC.shape[0]
        VnCount = self.matrixD.shape[1]

        return torch.zeros((V1Count, VnCount, VnCount//10 + 1, 1), device=self.device, dtype=torch.float32)

    def simulate(self, DF, param):
        if self.custom_batch_mode:
            self.step_custom(param)
        else:
            self.step(param)
        if param["mode"] == "fit":
            from utils import computeV2V4MSE
            combined_score, mse, spatial_metric = computeV2V4MSE(DF, self.matrixW)
            if "param" in param.get("output", "").lower():
                return combined_score
            return combined_score
        elif param["mode"] == "visualize":
            return None

    def step(self, param):
        mode = param["mode"]
        V1Count = self.matrixC.shape[0]
        VnCount = self.matrixD.shape[1]
        if mode == "fit":
            total_remaining = int(torch.sum(torch.diag(self.mask)).item())
            denom = max(1, int(self.batch_size_start + self.batch_size_end))
            est_steps = int(np.ceil(2 * max(1, total_remaining) / denom))
            if est_steps <= 0:
                est_steps = 1
            batch_idx = 0
            max_loops = total_remaining * 2 + 10
            with tqdm(total=total_remaining, desc="assign") as pbar:
                prev_remaining = total_remaining
                loops = 0
                while True:
                    remaining_now = int(torch.sum(torch.diag(self.mask)).item())
                    if remaining_now <= 0:
                        break
                    self.step_iter(V1Count, param["sampleMatrix"], batch_idx, est_steps)
                    batch_idx += 1
                    loops += 1
                    new_remaining = int(torch.sum(torch.diag(self.mask)).item())
                    assigned = max(0, prev_remaining - new_remaining)
                    if assigned > 0:
                        pbar.update(assigned)
                    prev_remaining = new_remaining
                    if loops >= max_loops:
                        break
        elif mode == "visualize":
            recordIdx = 0
            total_remaining = int(torch.sum(torch.diag(self.mask)).item())
            denom = max(1, int(self.batch_size_start + self.batch_size_end))
            est_steps = int(np.ceil(2 * max(1, total_remaining) / denom))
            if est_steps <= 0:
                est_steps = 1
            batch_idx = 0
            loops = 0
            while True:
                remaining_now = int(torch.sum(torch.diag(self.mask)).item())
                if remaining_now <= 0:
                    break
                self.step_iter(V1Count, param["sampleMatrix"], batch_idx, est_steps)
                batch_idx += 1
                if batch_idx % 10 == 0:
                    self.record[:, :, recordIdx, 0] = self.matrixW[:, V1Count:]
                    recordIdx += 1
                loops += 1
                if loops >= (total_remaining * 2 + 10):
                    break
            self.record[:, :, recordIdx, 0] = self.matrixW[:, V1Count:]

    def _assign_single_vn(self, best_col, temp, resource, V1Count, sampleMode, target_degree):
        """Assign V1 parents to a single Vn node."""
        v1_scores_col = temp[:, best_col]
        max_score = torch.max(v1_scores_col).item()

        connected_v1_indices = []

        if max_score <= 0.0 or target_degree <= 0:
            self.mask[best_col, best_col] = 0.0
            self.node_generation_order.append(best_col)
            return (best_col, [], np.array([0.0, 0.0]))

        k_parents = target_degree
        resource_col = resource[:, best_col] if resource.shape[1] > best_col else torch.ones_like(v1_scores_col)
        valid_mask = (v1_scores_col > 0.0) & (resource_col > 0.0)
        num_valid = int(torch.sum(valid_mask).item())

        if num_valid == 0:
            self.mask[best_col, best_col] = 0.0
            self.node_generation_order.append(best_col)
            return (best_col, [], np.array([0.0, 0.0]))

        actual_k = min(k_parents, num_valid)

        if sampleMode is not None and int(sampleMode) >= 0:
            scores_shift = v1_scores_col - torch.min(v1_scores_col[valid_mask])
            probs = scores_shift.clamp(min=0.0)
            probs[~valid_mask] = 0.0
            probs = probs / torch.sum(probs)
            selected_rows = torch.multinomial(probs, num_samples=actual_k, replacement=False)
        else:
            v1_scores_masked = v1_scores_col.clone()
            v1_scores_masked[~valid_mask] = -1e30
            _, selected_rows = torch.topk(v1_scores_masked, k=actual_k, largest=True)

        for r_idx in selected_rows:
            row = int(r_idx.item())
            self.matrixW[row, V1Count + best_col] += 1.0
            connected_v1_indices.append(row)

        if connected_v1_indices:
            sel = torch.tensor(connected_v1_indices, device=self.device, dtype=torch.long)
            c_sum = self.matrixC[:, sel].sum(dim=1)
            self._cached_propagation.add_(
                c_sum.unsqueeze(1) * self.matrixD[V1Count + best_col].unsqueeze(0)
            )
            self._cached_deg[sel] += 1.0

        if len(connected_v1_indices) > 0:
            v1_weights = self.matrixW[:, V1Count + best_col][connected_v1_indices]
            weight_sum = torch.sum(v1_weights)
            if weight_sum > 0:
                v1_weights_norm = v1_weights / weight_sum
                predicted_tuning = torch.sum(v1_weights_norm.unsqueeze(1) * self.V1_tuning_tensor[connected_v1_indices], dim=0)
                predicted_tuning_np = predicted_tuning.cpu().numpy()
            else:
                predicted_tuning_np = np.array([0.0, 0.0])
        else:
            predicted_tuning_np = np.array([0.0, 0.0])

        self.mask[best_col, best_col] = 0.0
        self.node_generation_order.append(best_col)
        return (best_col, connected_v1_indices, predicted_tuning_np)

    def step_iter(self, V1Count, sampleMode, batch_idx, total_batches):
        propagation = self._cached_propagation + self.direct_distance_weight * self.matrixD[:V1Count, :]

        resource = self.computeResource()

        temp = torch.multiply(propagation, resource)
        temp = temp @ self.mask

        VnCount = self.matrixD.shape[1]
        if VnCount == 0:
            return

        max_per_col = torch.max(temp, dim=0)[0]
        valid_vn_mask = (max_per_col > 0.0)
        valid_vn_count = int(torch.sum(valid_vn_mask).item())

        if valid_vn_count == 0:
            return

        best_v1_per_col = torch.argmax(temp, dim=0)
        col_indices = torch.arange(temp.shape[1], device=self.device)
        best_scores = temp[best_v1_per_col, col_indices]
        scores_masked = best_scores.clone()
        scores_masked[~valid_vn_mask] = -1e30

        target_degree = int(min(self.num_degree, V1Count)) if V1Count > 0 else 0

        progress = 0.0
        if total_batches > 0:
            progress = float(min(batch_idx + 1, total_batches)) / float(total_batches)
        if self.batch_size_schedule:
            schedule_idx = min(batch_idx, len(self.batch_size_schedule) - 1)
            current_bs = self.batch_size_schedule[schedule_idx]
        else:
            current_bs = int(round(self.batch_size_start + progress * (self.batch_size_end - self.batch_size_start)))
        if current_bs < 1:
            current_bs = 1

        if not self.batch_size_schedule and current_bs == 1:
            max_score_vn = torch.max(scores_masked).item()
            if max_score_vn <= -1e29:
                return
            eps = max(1e-6 * abs(max_score_vn), 1e-12)
            tied_mask = (scores_masked >= max_score_vn - eps)
            tied_cols = torch.where(tied_mask)[0]

            batch_nodes = []
            for col_idx in tied_cols:
                best_col = int(col_idx.item())
                entry = self._assign_single_vn(best_col, temp, resource, V1Count, sampleMode, target_degree)
                batch_nodes.append(entry)
            self.batch_info.append(batch_nodes)
            return

        pick_k = int(min(current_bs, valid_vn_count))
        _, topk_cols = torch.topk(scores_masked, k=pick_k, largest=True)
        candidate_cols = topk_cols.clone()

        batch_nodes = []

        for _ in range(pick_k):
            still_remaining_mask = (torch.diag(self.mask)[candidate_cols] > 0)
            if torch.sum(still_remaining_mask) == 0:
                break
            candidate_cols = candidate_cols[still_remaining_mask]

            propagation = self._cached_propagation + self.direct_distance_weight * self.matrixD[:V1Count, :]
            resource = self.computeResource()
            temp = torch.multiply(propagation, resource)
            temp = temp @ self.mask
            if candidate_cols.numel() == 1:
                best_col = int(candidate_cols[0].item())
            else:
                cand_mat = temp[:, candidate_cols]
                col_best_vals, _ = torch.max(cand_mat, dim=0)
                best_idx_in_cands = torch.argmax(col_best_vals)
                best_col = int(candidate_cols[best_idx_in_cands].item())

            entry = self._assign_single_vn(best_col, temp, resource, V1Count, sampleMode, target_degree)
            batch_nodes.append(entry)

        self.batch_info.append(batch_nodes)

    def step_custom(self, param):
        """Iterate pre-defined custom batches."""
        V1Count = self.matrixC.shape[0]
        VnCount = self.matrixD.shape[1]
        total_vn = len(self.custom_node_order)
        target_degree = int(min(self.num_degree, V1Count)) if V1Count > 0 else 0
        sampleMode = param["sampleMatrix"]

        print(f"[CustomBatch] Processing {total_vn} Vn nodes in {len(self.custom_batch_assignments)} batches")

        with tqdm(total=total_vn, desc="custom_assign") as pbar:
            for batch_id in sorted(self.custom_batch_assignments.keys()):
                batch_nodes_list = self.custom_batch_assignments[batch_id]
                batch_info_entries = []

                for df_idx in batch_nodes_list:
                    vn_col = df_idx - V1Count
                    if vn_col < 0 or vn_col >= VnCount:
                        continue

                    if self.mask[vn_col, vn_col].item() <= 0:
                        continue

                    indirect_propagation = self.matrixC @ self.matrixW @ self.matrixD
                    direct_propagation = self.matrixD[:V1Count, :]
                    propagation = indirect_propagation[:V1Count, :] + self.direct_distance_weight * direct_propagation
                    resource = self.computeResource()
                    temp = torch.multiply(propagation, resource)
                    temp = temp @ self.mask

                    v1_scores_col = temp[:, vn_col]
                    max_score = torch.max(v1_scores_col).item()

                    connected_v1_indices = []

                    if max_score <= 0.0 or target_degree <= 0:
                        self.mask[vn_col, vn_col] = 0.0
                        self.node_generation_order.append(vn_col)
                        batch_info_entries.append((vn_col, [], np.array([0.0, 0.0])))
                        pbar.update(1)
                        continue

                    resource_col = resource[:, vn_col] if resource.shape[1] > vn_col else torch.ones_like(v1_scores_col)
                    valid_mask = (v1_scores_col > 0.0) & (resource_col > 0.0)
                    num_valid = int(torch.sum(valid_mask).item())

                    if num_valid == 0:
                        self.mask[vn_col, vn_col] = 0.0
                        self.node_generation_order.append(vn_col)
                        batch_info_entries.append((vn_col, [], np.array([0.0, 0.0])))
                        pbar.update(1)
                        continue

                    actual_k = min(target_degree, num_valid)

                    if sampleMode is not None and int(sampleMode) >= 0:
                        scores_shift = v1_scores_col - torch.min(v1_scores_col[valid_mask])
                        probs = scores_shift.clamp(min=0.0)
                        probs[~valid_mask] = 0.0
                        probs = probs / torch.sum(probs)
                        selected_rows = torch.multinomial(probs, num_samples=actual_k, replacement=False)
                    else:
                        v1_scores_masked = v1_scores_col.clone()
                        v1_scores_masked[~valid_mask] = -1e30
                        _, selected_rows = torch.topk(v1_scores_masked, k=actual_k, largest=True)

                    for r_idx in selected_rows:
                        row = int(r_idx.item())
                        self.matrixW[row, V1Count + vn_col] += 1.0
                        connected_v1_indices.append(row)

                    if connected_v1_indices:
                        sel = torch.tensor(connected_v1_indices, device=self.device, dtype=torch.long)
                        c_sum = self.matrixC[:, sel].sum(dim=1)
                        self._cached_propagation.add_(
                            c_sum.unsqueeze(1) * self.matrixD[V1Count + vn_col].unsqueeze(0)
                        )
                        self._cached_deg[sel] += 1.0

                    if len(connected_v1_indices) > 0:
                        v1_weights = self.matrixW[:, V1Count + vn_col][connected_v1_indices]
                        weight_sum = torch.sum(v1_weights)
                        if weight_sum > 0:
                            v1_weights_norm = v1_weights / weight_sum
                            predicted_tuning = torch.sum(
                                v1_weights_norm.unsqueeze(1) * self.V1_tuning_tensor[connected_v1_indices], dim=0
                            )
                            predicted_tuning_np = predicted_tuning.cpu().numpy()
                        else:
                            predicted_tuning_np = np.array([0.0, 0.0])
                    else:
                        predicted_tuning_np = np.array([0.0, 0.0])

                    batch_info_entries.append((vn_col, connected_v1_indices, predicted_tuning_np))
                    self.mask[vn_col, vn_col] = 0.0
                    self.node_generation_order.append(vn_col)
                    pbar.update(1)

                self.batch_info.append(batch_info_entries)

    def sample(self, matrix, mode):
        if mode < 0:
            flat_idx = torch.argmax(matrix)
            return torch.unravel_index(flat_idx, matrix.shape)
        else:
            weights = matrix.flatten()
            weights = weights / torch.sum(weights)
            weights_cpu = weights.cpu().numpy()
            sample_idx = np.random.choice(len(weights_cpu), p=weights_cpu)
            return torch.unravel_index(torch.tensor(sample_idx, device=self.device), matrix.shape)

    def computeResource(self):
        V1Count = self.matrixC.shape[0]
        if V1Count == 0 or self.matrixD.shape[1] == 0:
            return torch.ones_like(self.matrixD[:V1Count, :], device=self.device, dtype=torch.float32)
        V1_resource = 1.0 / (self._cached_deg + 1.0)
        return V1_resource.unsqueeze(1).expand(-1, self.matrixD.shape[1])

