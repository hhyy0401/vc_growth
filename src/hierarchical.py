"""Hierarchical variant of the growth model (Supp. Fig. S7).

Growth starts from V1 exactly as in the main model. A stage ends once
STAGE_RATIO of that stage's source nodes have established at least one outgoing
connection; the targets assigned in the stage then become the sources of the
next stage, and the targets that are still unassigned stay targets. Each target
keeps the V1 node at the root of its chain, so the predicted tuning is read from
the same V1-to-target matrix as in the main model.
"""

import numpy as np
import torch
from tqdm import tqdm

from polarModel import KERNELS

STAGE_RATIO = 0.80      # fraction of a stage's sources that must connect to end it
IN_DEGREE_MAX = 1       # targets with at most this in-degree become next-stage sources
MAX_STAGES = 50


class HierarchicalMatrix(object):
    def __init__(self, dataDF, param):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_degree = int(param.get("num_degree", 1))
        self.radius = float(param.get("radius", 1.40))
        self.tangent = float(param.get("tangent", 2.20))
        self.tag = param.get("tag", None)

        self.N = int(dataDF.shape[0])
        self.V1Count = int(np.sum(dataDF["area"].values.astype(int) == 1))
        self.VnCount = self.N - self.V1Count

        self.kernel = KERNELS[param.get("kernel", "polar")](dataDF, self.radius, self.tangent, self.device)

        # V1 -> all nodes, as in the main model: [I | 0]
        self.matrixW = torch.cat(
            [
                torch.eye(self.V1Count, device=self.device, dtype=torch.float32),
                torch.zeros(self.V1Count, self.VnCount, device=self.device, dtype=torch.float32),
            ],
            dim=1,
        )
        self._cached_deg = torch.zeros(self.V1Count, device=self.device, dtype=torch.float32)

        V1_tuning = dataDF[dataDF["area"] == 1][["tuningX", "tuningY"]].values
        self.V1_tuning_tensor = torch.tensor(V1_tuning, device=self.device, dtype=torch.float32)
        self.node_generation_order = []
        self.batch_info = []

        self.simulate()

    def computeResource(self):
        """Competition among V1 sources: a source that already won targets is
        less available, 1 / (out-degree + 1)."""
        return (1.0 / (self._cached_deg + 1.0)).unsqueeze(1).expand(-1, self.VnCount)

    def simulate(self):
        sources = list(range(self.V1Count))                 # DF indices
        remaining_targets = list(range(self.V1Count, self.N))

        for stage_idx in range(MAX_STAGES):
            if not sources or not remaining_targets:
                break

            S, T = len(sources), len(remaining_targets)
            stage_nodes = sources + remaining_targets

            C_stage = self.kernel[sources, :][:, sources]                # (S, S)
            D_stage = self.kernel[stage_nodes, :][:, remaining_targets]  # (S+T, T)
            W_stage = torch.zeros((S, S + T), device=self.device, dtype=torch.float32)
            W_stage[:, :S] = torch.eye(S, device=self.device, dtype=torch.float32)

            target_alive = torch.ones((T,), device=self.device, dtype=torch.bool)
            target_assigned = torch.zeros((T,), device=self.device, dtype=torch.bool)
            target_in_degree = torch.zeros((T,), device=self.device, dtype=torch.int32)
            source_connected = torch.zeros((S,), device=self.device, dtype=torch.bool)

            direct = D_stage[:S, :]                                      # s(u,v), constant
            cached_indirect = C_stage @ W_stage @ D_stage

            max_loops = T * 2 + 10
            with tqdm(total=T, desc=f"hier_stage{stage_idx}") as pbar:
                prev_remaining, loops = T, 0
                while True:
                    if int(target_alive.sum().item()) <= 0:
                        break
                    if float(source_connected.float().mean().item()) >= STAGE_RATIO:
                        break

                    propagation = cached_indirect + direct
                    if stage_idx == 0:
                        vn_cols = torch.tensor([t - self.V1Count for t in remaining_targets],
                                               device=self.device, dtype=torch.long)
                        resource = self.computeResource()[:, vn_cols]
                    else:
                        stage_outdeg = W_stage[:, S:].sum(dim=1)
                        resource = (1.0 / (stage_outdeg + 1.0)).unsqueeze(1).expand(-1, T)

                    temp = propagation * resource
                    if not target_alive.all():
                        temp = temp.clone()
                        temp[:, ~target_alive] = -1e30

                    best_per_target = torch.max(temp, dim=0)[0]
                    valid_mask = (best_per_target > 0.0) & target_alive
                    if int(valid_mask.sum().item()) == 0:
                        break
                    scores_masked = best_per_target.clone()
                    scores_masked[~valid_mask] = -1e30

                    max_score = torch.max(scores_masked).item()
                    if max_score <= -1e29:
                        break
                    eps = max(1e-6 * abs(max_score), 1e-12)
                    tied_cols = torch.where(scores_masked >= max_score - eps)[0]

                    batch_nodes = []
                    for tc in tied_cols:
                        t_idx = int(tc.item())
                        batch_nodes.append(self._assign_one_target(
                            t_idx, temp, W_stage, S, sources, remaining_targets,
                            target_alive, target_assigned, target_in_degree, source_connected,
                        ))
                    for _t_idx, parents_stage_idx, _entry in batch_nodes:
                        for si in parents_stage_idx:
                            cached_indirect.add_(torch.outer(C_stage[:, si], D_stage[S + _t_idx, :]))
                    if batch_nodes:
                        self.batch_info.append([entry for _, _, entry in batch_nodes])

                    new_remaining = int(target_alive.sum().item())
                    assigned = max(0, prev_remaining - new_remaining)
                    if assigned > 0:
                        pbar.update(assigned)
                    prev_remaining = new_remaining
                    loops += 1
                    if loops >= max_loops:
                        break

            # Hand over: assigned targets become sources, unassigned ones stay targets.
            assigned_idx = torch.where(target_assigned)[0].tolist()
            unassigned_idx = torch.where(~target_assigned)[0].tolist()
            sources = [int(remaining_targets[ti]) for ti in assigned_idx
                       if int(target_in_degree[ti].item()) <= IN_DEGREE_MAX]
            remaining_targets = [int(remaining_targets[ti]) for ti in unassigned_idx]

    def _assign_one_target(self, t_idx, temp, W_stage, S, sources, remaining_targets,
                           target_alive, target_assigned, target_in_degree, source_connected):
        scores = temp[:, t_idx]
        parents_stage_idx = []
        if float(torch.max(scores).item()) > 0.0 and S > 0:
            k_parents = int(min(self.num_degree, S))
            _, pidx = torch.topk(scores, k=k_parents, largest=True)
            parents_stage_idx = [int(si) for si in pidx.tolist() if float(scores[si].item()) > 0.0]

        target_alive[t_idx] = False
        target_assigned[t_idx] = True
        for si in parents_stage_idx:
            W_stage[si, S + t_idx] += 1.0
            source_connected[si] = True
            target_in_degree[t_idx] += 1

        # Walk each parent back to the V1 node at the root of its chain.
        target_df = int(remaining_targets[t_idx])
        root_v1 = []
        for si in parents_stage_idx:
            parent_df = int(sources[si])
            rv1 = parent_df if parent_df < self.V1Count else int(torch.argmax(self.matrixW[:, parent_df]).item())
            self.matrixW[rv1, target_df] += 1.0
            self._cached_deg[rv1] += 1.0
            root_v1.append(rv1)

        self.node_generation_order.append(target_df - self.V1Count)
        if root_v1:
            predicted_tuning = self.V1_tuning_tensor[root_v1].mean(dim=0).cpu().numpy()
        else:
            predicted_tuning = np.array([0.0, 0.0])
        entry = (target_df - self.V1Count, root_v1, predicted_tuning)
        return t_idx, parents_stage_idx, entry
