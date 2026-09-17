from tqdm import tqdm
import torch
import numpy as np
import sys
sys.path.insert(0, '..')
from TUNING_COLOR_UTILS import compute_tuning_colors


def polar_kernel(DF, radius, tangent, device):
    """Activity-correlation kernel s(u,v) on the 2D embedding.

    At each source node i the displacement to j is decomposed in i's own frame,
    radial (away from the fovea) and tangential, and scaled by sigma_R and
    sigma_T, so the kernel is anisotropic and asymmetric.
    """
    r_coords = torch.tensor(DF["r"].values, device=device, dtype=torch.float32)
    t_coords = torch.tensor(DF["t"].values, device=device, dtype=torch.float32)

    vx = r_coords * torch.cos(t_coords)
    vy = r_coords * torch.sin(t_coords)

    dx = vx.unsqueeze(0) - vx.unsqueeze(1)   # dx[i,j] = vx_j - vx_i
    dy = vy.unsqueeze(0) - vy.unsqueeze(1)

    cos_t = torch.cos(t_coords).unsqueeze(1)
    sin_t = torch.sin(t_coords).unsqueeze(1)

    d_r = dx * cos_t + dy * sin_t
    d_t = -dx * sin_t + dy * cos_t

    d = torch.sqrt((d_r / float(radius))**2 + (d_t / float(tangent))**2)
    return torch.exp(-d)


def sphere_geodesic_kernel(DF, radius, tangent, device):
    """Kernel from geodesic distances on the smooth spherical surface (Supp. Fig. S6).

    For a source node i and a target j, the geodesic g_ij on the sphere is
    decomposed in i's own tangent plane (Riemannian log map) into a radial
    component, away from the foveal node, and a tangential one, which are then
    scaled by sigma_R and sigma_T exactly as in :func:`polar_kernel`. Geodesic
    distances are rescaled so that the largest pairwise distance matches the
    largest pairwise distance of the MDS embedding, which puts sigma on the same
    scale in both substrates.
    """
    R_SPHERE = 100.0
    P = torch.stack([
        torch.tensor(DF[c].values, device=device, dtype=torch.float64) for c in ("sx", "sy", "sz")
    ], dim=1)
    U = P / torch.linalg.norm(P, dim=1, keepdim=True)          # unit vectors (N, 3)

    center_flags = torch.tensor(DF["is_center"].values, device=device, dtype=torch.long)
    uc = U[int(torch.where(center_flags == 1)[0][0].item())]   # foveal node

    # Local frame at each source: e_r points away from the fovea, e_t is
    # perpendicular to it in the same tangent plane.
    dotc = (U @ uc).unsqueeze(1)
    er = U * dotc - uc.unsqueeze(0)
    er_norm = torch.linalg.norm(er, dim=1, keepdim=True)
    er = er / torch.clamp(er_norm, min=1e-12)
    degenerate = (er_norm.squeeze(1) < 1e-9)                   # only at the pole itself
    if degenerate.any():
        ax = torch.zeros_like(U)
        ax[torch.arange(len(U), device=device), torch.argmin(torch.abs(U), dim=1)] = 1.0
        alt = ax - (ax * U).sum(1, keepdim=True) * U
        alt = alt / torch.clamp(torch.linalg.norm(alt, dim=1, keepdim=True), min=1e-12)
        er = torch.where(degenerate.unsqueeze(1), alt, er)
    et = torch.linalg.cross(U, er)
    et = et / torch.clamp(torch.linalg.norm(et, dim=1, keepdim=True), min=1e-12)

    cos_ij = torch.clamp(U @ U.t(), -1.0, 1.0)
    sin_ij = torch.clamp(torch.sqrt(torch.clamp(1.0 - cos_ij ** 2, min=0.0)), min=1e-9)
    scale = (R_SPHERE * torch.arccos(cos_ij)) / sin_ij
    del cos_ij
    d_r = scale * (er @ U.t())
    d_t = scale * (et @ U.t())
    del scale, sin_ij

    # Put the sphere on the scale of the MDS embedding.
    mds_xy = torch.stack([
        torch.tensor(DF["x"].values, device=device, dtype=torch.float64),
        torch.tensor(DF["y"].values, device=device, dtype=torch.float64),
    ], dim=1)
    norm = torch.cdist(mds_xy, mds_xy).max() / torch.sqrt(d_r ** 2 + d_t ** 2).max()
    d_r, d_t = d_r * norm, d_t * norm

    d = torch.sqrt((d_r / float(radius)) ** 2 + (d_t / float(tangent)) ** 2)
    return torch.exp(-d).to(torch.float32)


KERNELS = {"polar": polar_kernel, "sphere": sphere_geodesic_kernel}


class VisualMatrix3D(object):
    def __init__(self, dataDF, param):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_degree = int(param.get("num_degree", 1))
        self.radius = float(param.get("radius", 1.30))
        self.tangent = float(param.get("tangent", 2.20))
        self.tag = param.get("tag", None)
        self.kernel_type = param.get("kernel", "polar")

        # Optional spatially defined growth order (Supp. Fig. S4).
        self.custom_batch_mode = param.get("custom_batch_mode", None)
        self.custom_node_order = None
        self.custom_batch_assignments = None
        if self.custom_batch_mode:
            from custom_batch import parse_custom_mode, get_custom_node_order
            mode_name, ecc_order = parse_custom_mode(self.custom_batch_mode)
            print(f"Custom batch mode: {self.custom_batch_mode} (mode={mode_name}, ecc={ecc_order})")
            self.custom_node_order, self.custom_batch_assignments = get_custom_node_order(
                dataDF, mode_name, ecc_order, n_batches=30,
                radius=self.radius, tangent_deg=self.tangent,
            )

        self.matrixC, self.matrixW, self.matrixD, self.mask = self.initMatrix(
            dataDF, radius=self.radius, tangent=self.tangent,
        )

        # Activity correlation c(u,v) = s(u,v) + sum over existing edges (i,j) of
        # s(u,i) s(j,v). The direct term s(u,v) is constant; the indirect term is
        # C @ W @ D, which starts at C @ D because W = [I | 0] and is updated
        # incrementally as edges are added.
        _V1 = self.matrixC.shape[0]
        self._cached_propagation = (self.matrixC @ self.matrixD[:_V1, :]).clone()
        self._cached_propagation = self._cached_propagation + self.matrixD[:_V1, :]
        self._cached_deg = torch.zeros(_V1, device=self.device, dtype=torch.float32)

        V1_tuning = dataDF[dataDF["area"] == 1][["tuningX", "tuningY"]].values
        self.V1_tuning_tensor = torch.tensor(V1_tuning, device=self.device, dtype=torch.float32)
        self.node_generation_order = []
        self.batch_info = []
        self.simulate()

    def initMatrix(self, DF, radius=1.30, tangent=2.20):
        V1Count = DF[(DF["area"] == 1)].shape[0]
        VnCount = DF[(DF["area"] != 1)].shape[0]

        mask = torch.eye(VnCount, device=self.device, dtype=torch.float32)
        kernel = KERNELS[self.kernel_type](DF, radius, tangent, self.device)

        matrixC = kernel[:V1Count, :V1Count]
        matrixD = kernel[:, V1Count:]
        matrixW = torch.cat([torch.eye(V1Count, device=self.device), torch.zeros(V1Count, VnCount, device=self.device)], dim=1)
        return matrixC, matrixW, matrixD, mask

    def simulate(self):
        if self.custom_batch_mode:
            self.step_custom()
        else:
            self.step()

    def step(self):
        V1Count = self.matrixC.shape[0]
        total_remaining = int(torch.sum(torch.diag(self.mask)).item())
        max_loops = total_remaining * 2 + 10  # safety bound
        with tqdm(total=total_remaining, desc="assign") as pbar:
            prev_remaining = total_remaining
            loops = 0
            while True:
                if int(torch.sum(torch.diag(self.mask)).item()) <= 0:
                    break
                self.step_iter(V1Count)
                loops += 1
                new_remaining = int(torch.sum(torch.diag(self.mask)).item())
                assigned = max(0, prev_remaining - new_remaining)
                if assigned > 0:
                    pbar.update(assigned)
                prev_remaining = new_remaining
                if loops >= max_loops:
                    break

    def _assign_single_vn(self, best_col, temp, resource, V1Count, target_degree):
        """Assign V1 parents to a single Vn node using the given temp matrix.
        Returns (best_col, connected_v1_indices, predicted_tuning_np)."""
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

    def step_iter(self, V1Count):
        propagation = self._cached_propagation

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

        # Assign every Vn sharing the current maximum score simultaneously, from the
        # same temp matrix (no recomputation between tied nodes).
        max_score_vn = torch.max(scores_masked).item()
        if max_score_vn <= -1e29:
            return
        eps = max(1e-6 * abs(max_score_vn), 1e-12)
        tied_cols = torch.where(scores_masked >= max_score_vn - eps)[0]

        batch_nodes = []
        for col_idx in tied_cols:
            best_col = int(col_idx.item())
            batch_nodes.append(
                self._assign_single_vn(best_col, temp, resource, V1Count, target_degree)
            )
        self.batch_info.append(batch_nodes)

    def step_custom(self):
        """Custom batch mode: iterate pre-defined batches/nodes.
        Node order is fully pre-defined; edge computation uses same W matrix logic."""
        V1Count = self.matrixC.shape[0]
        VnCount = self.matrixD.shape[1]
        total_vn = len(self.custom_node_order)
        target_degree = int(min(self.num_degree, V1Count)) if V1Count > 0 else 0

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

                    # c(u,v) = s(u,v) + sum over existing edges, as in step_iter.
                    indirect_propagation = self.matrixC @ self.matrixW @ self.matrixD
                    propagation = indirect_propagation[:V1Count, :] + self.matrixD[:V1Count, :]
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

    def computeResource(self):
        V1Count = self.matrixC.shape[0]
        if V1Count == 0 or self.matrixD.shape[1] == 0:
            return torch.ones_like(self.matrixD[:V1Count, :], device=self.device, dtype=torch.float32)
        V1_resource = 1.0 / (self._cached_deg + 1.0)
        return V1_resource.unsqueeze(1).expand(-1, self.matrixD.shape[1])
