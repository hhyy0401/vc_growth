from tqdm import tqdm
import torch
import numpy as np
from utils import initDirectory

class VisualMatrix3D(object):
    def __init__(self, dataDF, param, outputDir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.outputDir = initDirectory(param, outputDir)

        # Core params
        self.num_degree = int(param.get("num_degree", 1))
        self.alpha = float(param.get("alpha", 0.4))
        self.mode = param.get("coordinate_mode", "sphere")
        self.batch_size_start = int(param.get("batch_size_start", int(param.get("batch_size", 1))))
        self.batch_size_end = int(param.get("batch_size_end", int(param.get("batch_size", 1))))

        self.use_dynamic_batch_size = bool(param.get("use_dynamic_batch_size", False))
        self.batch_size_schedule = []

        if self.use_dynamic_batch_size:
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
        else:
            self.batch_size_start = int(param.get("batch_size_start", 1))
            self.batch_size_end = int(param.get("batch_size_end", 1))

        self.radius = float(param.get("radius", 2.0))
        self.tangent = float(param.get("tangent", 2.0))
        self.distance_mode = param.get("distance_mode", "polar")
        self.direct_distance_weight = 1.0
        self.tag = param.get("tag", None)

        # Hierarchical params
        self.hierarchical = bool(param.get("hierarchical", False))
        self.stage_ratio = float(param.get("stage_ratio", 0.90))      # e.g. 0.9
        self.in_degree_max = int(param.get("in_degree_max", 1))       # e.g. 1
        self.max_stages = int(param.get("max_stages", 50))

        # Data
        self.dataDF = dataDF
        self.N = int(dataDF.shape[0])
        self.v1_mask = (dataDF["area"].values.astype(int) == 1)
        self.V1Count = int(np.sum(self.v1_mask))
        self.VnCount = int(self.N - self.V1Count)

        # Full kernel (N,N) once
        self.kernel = self._build_full_kernel(
            dataDF,
            radius=self.radius,
            tangent=self.tangent,
            distance_mode=self.distance_mode,
        )

        self.matrixC = self.kernel[: self.V1Count, : self.V1Count]              # (V1,V1)
        self.matrixD = self.kernel[:, self.V1Count :]                            # (N,Vn)
        self.mask = torch.eye(self.VnCount, device=self.device, dtype=torch.float32)

        # Global W: (V1, N) = [I | 0]
        self.matrixW = torch.cat(
            [
                torch.eye(self.V1Count, device=self.device, dtype=torch.float32),
                torch.zeros(self.V1Count, self.VnCount, device=self.device, dtype=torch.float32),
            ],
            dim=1,
        )
        # Cached degree for computeResource (updated incrementally)
        self._cached_deg = torch.zeros(self.V1Count, device=self.device, dtype=torch.float32)

        # Logs for video/debug
        self.node_generation_order = []  # global DF indices
        self.batch_info = []             # list of batches; each batch list of dicts

        self.indicator = self.simulate(dataDF, param)

    def _build_full_kernel(self, DF, radius=2.0, tangent=2.0, distance_mode="polar"):
        radius = float(radius)
        tangent = float(tangent)

        if distance_mode == "euclidean":
            x_coords = torch.tensor(DF["x"].values, device=self.device, dtype=torch.float32)
            y_coords = torch.tensor(DF["y"].values, device=self.device, dtype=torch.float32)
            z_coords = torch.tensor(DF["z"].values, device=self.device, dtype=torch.float32)
            x1, x2 = x_coords.unsqueeze(1), x_coords.unsqueeze(0)
            y1, y2 = y_coords.unsqueeze(1), y_coords.unsqueeze(0)
            z1, z2 = z_coords.unsqueeze(1), z_coords.unsqueeze(0)
            euclidean_distance = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            kernel = torch.exp(-euclidean_distance / radius)
        elif distance_mode == "arc":
            if "r" in DF.columns:
                r_coords = torch.tensor(DF["r"].values, device=self.device, dtype=torch.float32)
            else:
                x_coords = torch.tensor(DF["x"].values, device=self.device, dtype=torch.float32)
                y_coords = torch.tensor(DF["y"].values, device=self.device, dtype=torch.float32)
                r_coords = torch.sqrt(x_coords**2 + y_coords**2)
            if "t" in DF.columns:
                t_coords = torch.tensor(DF["t"].values, device=self.device, dtype=torch.float32)
            else:
                x_coords = torch.tensor(DF["x"].values, device=self.device, dtype=torch.float32)
                y_coords = torch.tensor(DF["y"].values, device=self.device, dtype=torch.float32)
                t_coords = torch.atan2(y_coords, x_coords)

            r1 = r_coords.unsqueeze(1)
            r2 = r_coords.unsqueeze(0)
            t1 = t_coords.unsqueeze(1)
            t2 = t_coords.unsqueeze(0)

            theta_diff = t1 - t2
            angle_distance = torch.minimum(torch.abs(theta_diff), torch.abs(torch.abs(theta_diff) - 2 * np.pi))
            radius_distance = torch.abs(r1 - r2)

            r_min = torch.minimum(r1, r2)
            arc_distance = r_min * angle_distance
            tangent_rad = tangent * np.pi / 180.0
            kernel_tangent = torch.exp(-arc_distance / tangent_rad)
            kernel_euclidean = torch.exp(-radius_distance / radius)
            kernel = kernel_euclidean * kernel_tangent
        else:
            # At each source point i, define local basis from its polar angle:
            #   radial     = (cos t_i, sin t_i)   — direction away from fovea
            #   tangential = (-sin t_i, cos t_i)   — perpendicular
            # Displacement (i->j) is decomposed in i's basis (asymmetric).
            r_coords = torch.tensor(DF["r"].values, device=self.device, dtype=torch.float32)
            t_coords = torch.tensor(DF["t"].values, device=self.device, dtype=torch.float32)

            vx = r_coords * torch.cos(t_coords)
            vy = r_coords * torch.sin(t_coords)

            dx = vx.unsqueeze(0) - vx.unsqueeze(1)  # (N, N), dx[i,j] = vx_j - vx_i
            dy = vy.unsqueeze(0) - vy.unsqueeze(1)

            cos_t = torch.cos(t_coords).unsqueeze(1)  # (N, 1)
            sin_t = torch.sin(t_coords).unsqueeze(1)

            d_r = dx * cos_t + dy * sin_t       # radial component
            d_t = -dx * sin_t + dy * cos_t      # tangential component

            d = torch.sqrt((d_r / radius)**2 + (d_t / tangent)**2)
            kernel = torch.exp(-d)

        return kernel  # (N,N)

    def computeResource(self):
        if self.V1Count == 0 or self.VnCount == 0:
            return torch.ones((self.V1Count, max(1, self.VnCount)), device=self.device, dtype=torch.float32)
        # Use cached degree (updated incrementally in _assign_one_target)
        V1_resource = 1.0 / (self._cached_deg + 1.0)
        return V1_resource.unsqueeze(1).expand(-1, self.VnCount)  # (V1,Vn)

    def simulate(self, DF, param):
        if self.hierarchical:
            self.step_hierarchical(param)
        else:
            # fallback: original single-stage behavior
            self.step_baseline(param)

        if param.get("mode", "fit") == "fit":
            from utils import computeV2V4MSE
            combined_score, _, _ = computeV2V4MSE(DF, self.matrixW)
            return combined_score
        return None

    def step_baseline(self, param):
        if not self.use_dynamic_batch_size:
            self.batch_size_start = 1
            self.batch_size_end = 1

        self.hierarchical = True
        self.stage_ratio = 1.0  # run until targets exhausted
        self.max_stages = 1
        self.step_hierarchical(param)
        self.hierarchical = False

    def step_hierarchical(self, param):
        if self.V1Count == 0 or self.VnCount == 0:
            return
        if not (0.0 < self.stage_ratio <= 1.0):
            raise ValueError(f"stage_ratio must be in (0,1] (got {self.stage_ratio})")
        if self.in_degree_max < 1:
            raise ValueError(f"in_degree_max must be >=1 (got {self.in_degree_max})")

        # stage0
        sources = list(range(self.V1Count))              # DF indices
        remaining_targets = list(range(self.V1Count, self.N))

        stage_idx = 0
        while stage_idx < self.max_stages:
            if len(sources) == 0 or len(remaining_targets) == 0:
                break

            S = len(sources)
            T = len(remaining_targets)

            stage_nodes = sources + remaining_targets  # length S+T

            # Stage matrices
            C_stage = self.kernel[sources, :][:, sources]                      # (S,S)
            D_stage = self.kernel[stage_nodes, :][:, remaining_targets]        # (S+T, T)

            # W_stage: (S, S+T), init [I | 0]
            W_stage = torch.zeros((S, S + T), device=self.device, dtype=torch.float32)
            W_stage[:, :S] = torch.eye(S, device=self.device, dtype=torch.float32)

            # Target availability within stage
            target_alive = torch.ones((T,), device=self.device, dtype=torch.bool)

            # Degree bookkeeping (stage-local)
            source_connected = torch.zeros((S,), device=self.device, dtype=torch.bool)  # outdeg>=1?
            target_in_degree = torch.zeros((T,), device=self.device, dtype=torch.int32)
            target_assigned = torch.zeros((T,), device=self.device, dtype=torch.bool)

            # Estimate loop counts for batch-size interpolation
            total_remaining = int(target_alive.sum().item())
            denom = max(1, int(self.batch_size_start + self.batch_size_end))
            est_steps = int(np.ceil(2 * max(1, total_remaining) / denom))
            est_steps = max(1, est_steps)
            batch_idx = 0
            max_loops = total_remaining * 2 + 10

            direct = D_stage[:S, :]  # (S,T) — constant throughout stage
            cached_indirect = C_stage @ W_stage @ D_stage  # (S,T)

            with tqdm(total=total_remaining, desc=f"hier_stage{stage_idx}") as pbar:
                prev_remaining = total_remaining
                loops = 0

                while True:
                    remaining_now = int(target_alive.sum().item())
                    if remaining_now <= 0:
                        break

                    # Stop stage if >= stage_ratio of sources connected
                    connected_ratio = float(source_connected.float().mean().item()) if S > 0 else 1.0
                    if connected_ratio >= self.stage_ratio:
                        break

                    # Propagation (use cached indirect — rank-1 updated)
                    propagation = cached_indirect + self.direct_distance_weight * direct

                    # Resource
                    if stage_idx == 0:
                        # resource columns correspond to Vn columns (df_idx - V1Count)
                        res_v1 = self.computeResource()  # (V1,Vn)
                        vn_cols = torch.tensor([t - self.V1Count for t in remaining_targets], device=self.device, dtype=torch.long)
                        resource = res_v1[:, vn_cols]     # (S,T)
                    else:
                        # Stage 1+: compute resource from W_stage out-degree
                        stage_outdeg = W_stage[:, S:].sum(dim=1)  # (S,)
                        deg_res = 1.0 / (stage_outdeg + 1.0)
                        resource = deg_res.unsqueeze(1).expand(-1, T)  # (S,T)

                    temp = propagation * resource
                    if not target_alive.all():
                        temp = temp.clone()
                        temp[:, ~target_alive] = -1e30

                    # Valid targets (any positive score)
                    max_per_col = torch.max(temp, dim=0)[0]
                    valid_mask = (max_per_col > 0.0) & target_alive
                    valid_count = int(valid_mask.sum().item())
                    if valid_count == 0:
                        break

                    # Batch size
                    progress = float(min(batch_idx + 1, est_steps)) / float(est_steps)
                    if self.batch_size_schedule:
                        schedule_idx = min(batch_idx, len(self.batch_size_schedule) - 1)
                        current_bs = self.batch_size_schedule[schedule_idx]
                    else:
                        current_bs = int(round(self.batch_size_start + progress * (self.batch_size_end - self.batch_size_start)))

                    current_bs = max(1, current_bs)

                    # Choose targets by best score
                    best_src_per_t = torch.argmax(temp, dim=0)
                    cols = torch.arange(T, device=self.device)
                    best_scores = temp[best_src_per_t, cols]
                    best_scores_masked = best_scores.clone()
                    best_scores_masked[~valid_mask] = -1e30

                    def _assign_one_target(t_idx, temp_mat):
                        scores = temp_mat[:, t_idx]
                        ms = float(torch.max(scores).item())
                        parents_s = []
                        if ms > 0.0 and S > 0:
                            k_par = int(min(self.num_degree, S))
                            if int(param.get("sampleMatrix", -1)) == -1:
                                _, pidx = torch.topk(scores, k=k_par, largest=True)
                            else:
                                probs = torch.softmax(scores, dim=0)
                                pidx = torch.multinomial(probs, num_samples=k_par, replacement=False)
                            for si in pidx.tolist():
                                if float(scores[si].item()) <= 0.0:
                                    continue
                                parents_s.append(int(si))

                        target_alive[t_idx] = False
                        target_assigned[t_idx] = True
                        for si in parents_s:
                            W_stage[si, S + t_idx] += 1.0
                            source_connected[si] = True
                            target_in_degree[t_idx] += 1

                        tgt_df = int(remaining_targets[t_idx])
                        par_df = [int(sources[s]) for s in parents_s]
                        rts = []
                        for pdf in par_df:
                            if pdf < self.V1Count:
                                rv1 = pdf
                            else:
                                rv1 = int(torch.argmax(self.matrixW[:, pdf]).item())
                            self.matrixW[rv1, tgt_df] += 1.0
                            self._cached_deg[rv1] += 1.0
                            rts.append(rv1)

                        self.node_generation_order.append(tgt_df)
                        return {
                            "stage": int(stage_idx),
                            "target_df_idx": tgt_df,
                            "parents_df_idx": par_df,
                            "parents_stage_idx": parents_s,
                            "target_stage_idx": t_idx,
                            "root_v1_idx": int(rts[0]) if rts else -1,
                        }

                    batch_nodes = []

                    if not self.batch_size_schedule and current_bs == 1:
                        max_score_t = torch.max(best_scores_masked).item()
                        if max_score_t <= -1e29:
                            batch_idx += 1; loops += 1; continue
                        eps = max(1e-6 * abs(max_score_t), 1e-12)
                        tied_mask = (best_scores_masked >= max_score_t - eps)
                        tied_cols = torch.where(tied_mask)[0]

                        for tc in tied_cols:
                            t_idx = int(tc.item())
                            entry = _assign_one_target(t_idx, temp)
                            batch_nodes.append(entry)

                        # Incremental rank-1 update to cached_indirect
                        for entry in batch_nodes:
                            ti = entry["target_stage_idx"]
                            for si in entry["parents_stage_idx"]:
                                cached_indirect.add_(
                                    torch.outer(C_stage[:, si], D_stage[S + ti, :])
                                )
                    else:
                        pick_k = int(min(current_bs, valid_count))
                        _, topk_t = torch.topk(best_scores_masked, k=pick_k, largest=True)

                        for _ in range(pick_k):
                            still = target_alive[topk_t]
                            if int(still.sum().item()) == 0:
                                break
                            topk_t = topk_t[still]

                            # Use cached_indirect (updated incrementally below)
                            propagation = cached_indirect + self.direct_distance_weight * direct
                            if stage_idx == 0:
                                res_v1 = self.computeResource()
                                vn_cols_r = torch.tensor([t - self.V1Count for t in remaining_targets], device=self.device, dtype=torch.long)
                                resource = res_v1[:, vn_cols_r]
                            else:
                                stage_outdeg = W_stage[:, S:].sum(dim=1)
                                deg_res = 1.0 / (stage_outdeg + 1.0)
                                resource = deg_res.unsqueeze(1).expand(-1, T)

                            temp = propagation * resource
                            if not target_alive.all():
                                temp = temp.clone()
                                temp[:, ~target_alive] = -1e30

                            if topk_t.numel() == 1:
                                t_idx = int(topk_t[0].item())
                            else:
                                cand_mat = temp[:, topk_t]
                                cand_best, _ = torch.max(cand_mat, dim=0)
                                best_in_cands = torch.argmax(cand_best)
                                t_idx = int(topk_t[best_in_cands].item())

                            entry = _assign_one_target(t_idx, temp)
                            # Incremental rank-1 update to cached_indirect
                            ti = entry["target_stage_idx"]
                            for si in entry["parents_stage_idx"]:
                                cached_indirect.add_(
                                    torch.outer(C_stage[:, si], D_stage[S + ti, :])
                                )
                            batch_nodes.append(entry)

                    if batch_nodes:
                        self.batch_info.append(batch_nodes)

                    batch_idx += 1
                    loops += 1

                    new_remaining = int(target_alive.sum().item())
                    assigned_now = max(0, prev_remaining - new_remaining)
                    if assigned_now > 0:
                        pbar.update(assigned_now)
                    prev_remaining = new_remaining

                    if loops >= max_loops:
                        break

            # Stage ends: next sources/targets
            assigned_idx = torch.where(target_assigned)[0].tolist()
            unassigned_idx = torch.where(~target_assigned)[0].tolist()

            next_sources = []
            for ti in assigned_idx:
                if int(target_in_degree[ti].item()) <= self.in_degree_max:
                    next_sources.append(int(remaining_targets[ti]))  # DF idx

            next_targets = [int(remaining_targets[ti]) for ti in unassigned_idx]

            # source that never connected is discarded automatically (not carried)
            sources = next_sources
            remaining_targets = next_targets
            stage_idx += 1
