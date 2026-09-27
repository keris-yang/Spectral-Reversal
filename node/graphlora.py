"""
GraphLoRA and GraphLoFT adapters for node-level GNN (GCN backbone).

GraphLoRA (Wei et al., 2024) — faithful adaptation to the EdgePrompt framework.

    Two modules (node feature projection step removed):

    1. Structural Knowledge Transfer (frozen GCN + LoRA branch, separated at
       the last layer for GCT contrastive learning):
           z_frozen = GCNConv_frozen(h)
           z_lora   = GCN-aggregated LoRA correction
           z        = z_frozen + z_lora

       GCT loss: symmetric InfoNCE with class-aware positive mask between
       z_frozen and z_lora (from batched_gct_loss in GraphLoRA/util.py).

    2. Structure-aware Regularization (adjacency reconstruction):
           rec_adj[i,j] = σ( softmax(logits)[i] · softmax(logits)[j]^T )
           L_rec = weighted-BCE(rec_adj, A_subgraph)

    Combined training loss:
        L = cls_loss + ct_weight · L_ct + rec_weight · L_rec

GraphLoFT (proposed) — GraphLoRA architecture + LoFT training dynamics:

    Extends GraphLoRA with LoFT (Tastan et al., ICLR 2026):
      https://arxiv.org/abs/2505.21289

    • Node feature projection step removed; two remaining modules (LoRA
      knowledge transfer, structure-aware reg) share the same combined loss.
    • Parameters use LoFT's standard convention (A=[r,d_in], B=[d_out,r])
      so that LoFTAdamW's gradient rescaling and momentum reprojection apply.
    • Trained with LoFTAdamW instead of Adam, making the LoRA weight
      trajectory match full fine-tuning (FFT).

    LoFT gradient rescaling removes the LoRA parameterisation artifact:
        g̃_A = (B.T @ B + εI)⁻¹ @ g_A     [for A=[r,d_in]]
        g̃_B = g_B @ (A @ A.T + εI)⁻¹     [for B=[d_out,r]]
    Combined with momentum reprojection and row-product second moments,
    the optimizer state tracks the same trajectory as FFT-AdamW.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch_geometric.utils import add_self_loops, degree, to_dense_adj


# ---------------------------------------------------------------------------
# GraphLoRA  (GCN / node classification)
# ---------------------------------------------------------------------------

class GraphLoRA(nn.Module):
    """Full GraphLoRA for a frozen 2-layer GCN (node classification).

    Two modules (LoRA knowledge transfer + structure-aware reg).
    Node feature projector removed: not applicable to same-graph setting.
    Parameters: lora_A=[d_in, r], lora_B=[r, d_out]  (GraphLoRA convention).
    Trained with standard Adam; use GraphLoFT for LoFT-style optimizer.
    """

    def __init__(
        self,
        dim_in_list: List[int],
        dim_out_list: List[int],
        r: int = 8,
        ct_weight: float = 0.5,
        rec_weight: float = 0.1,
    ):
        super().__init__()
        assert len(dim_in_list) == len(dim_out_list)
        self.num_layers = len(dim_in_list)
        self.r = r
        self.ct_weight  = ct_weight
        self.rec_weight = rec_weight

        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for d_in, d_out in zip(dim_in_list, dim_out_list):
            A = nn.Parameter(torch.empty(d_in, r))
            B = nn.Parameter(torch.zeros(r, d_out))
            nn.init.kaiming_normal_(A)
            self.lora_A.append(A)
            self.lora_B.append(B)

    def get_prompt(self, h: torch.Tensor, edge_index: torch.Tensor,
                   layer: int) -> torch.Tensor:
        """GCN-normalised LoRA correction.

        h_down = h @ A        [N, r]   (down-project)
        h_agg  = D^{-1/2}-aggregate(h_down)
        p      = h_agg @ B   [N, d_out]  (up-project)
        """
        A = self.lora_A[layer]   # [d_in, r]
        B = self.lora_B[layer]   # [r, d_out]
        h_down = h @ A

        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        row, col = edge_index_sl
        deg = degree(col, num_nodes=h.size(0), dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h_msg = h_down[col] * norm.unsqueeze(1)
        h_agg = torch.zeros_like(h_down)
        h_agg.scatter_add_(0, row.unsqueeze(1).expand_as(h_msg), h_msg)
        return h_agg @ B

    # ------ shared static losses (used by both GraphLoRA and GraphLoFT) ------

    @staticmethod
    def gct_loss(
        z_frozen: torch.Tensor,
        z_lora: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        tau: float = 0.5,
        sup_weight: float = 0.2,
    ) -> torch.Tensor:
        """Symmetric GCT contrastive loss (batched_gct_loss from GraphLoRA)."""
        N = z_frozen.size(0)
        if N < 2:
            return torch.tensor(0.0, device=z_frozen.device, requires_grad=True)
        z1 = F.normalize(z_frozen, dim=-1)
        z2 = F.normalize(z_lora,   dim=-1)
        if labels is not None and labels.numel() == N:
            y = labels.view(-1, 1).long()
            mask = sup_weight * (y == y.T).float()
            mask.fill_diagonal_(1.0)
        else:
            mask = torch.eye(N, device=z1.device)

        def _one_dir(za, zb):
            refl  = torch.exp(za @ za.T / tau)
            cross = torch.exp(za @ zb.T / tau)
            numer = (mask * cross).sum(1)
            denom = refl.sum(1) + cross.sum(1) - refl.diag()
            return -torch.log(numer / denom.clamp(min=1e-10)).mean()

        return 0.5 * (_one_dir(z1, z2) + _one_dir(z2, z1))

    @staticmethod
    def reconstruction_loss(
        logits_per_node: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Adjacency reconstruction from per-node class logits."""
        if num_nodes > 300:
            return torch.tensor(0.0, device=logits_per_node.device)
        adj     = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        probs   = torch.softmax(logits_per_node, dim=-1)
        rec_adj = torch.sigmoid(probs @ probs.T)
        num_pos   = adj.sum().clamp(min=1)
        pos_weight = (num_nodes ** 2 - num_pos) / num_pos / 10.0
        weight = torch.ones(num_nodes ** 2, device=logits_per_node.device)
        weight[adj.view(-1) == 1] = pos_weight
        return F.binary_cross_entropy(rec_adj.view(-1), adj.view(-1),
                                      weight=weight)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        dims = [(a.shape[0], b.shape[1]) for a, b in zip(self.lora_A, self.lora_B)]
        return (
            f"GraphLoRA(r={self.r}, layers={dims}, "
            f"ct={self.ct_weight}, rec={self.rec_weight}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# GraphLoFT  (GCN / node classification)
# ---------------------------------------------------------------------------

class GraphLoFT(nn.Module):
    """GraphLoFT — LoFT-style LoRA correction (GCN).

    Node feature projector removed: not applicable to same-graph setting.
    Intentionally omits GCT contrastive loss and adjacency reconstruction
    loss (modules 2b and 3 in the GraphLoRA paper).

    Design rationale — why GCT/rec are dropped:
      LoFT's gradient rescaling (g̃_A = (B.T@B + εI)⁻¹ @ g_A, etc.) is
      derived assuming gradient signals are dominated by a single objective.
      The GCT contrastive loss produces ∂L_ct/∂B ∝ (h @ A.T)^T, whose norm
      grows with ||A||.  When combined with LoFT's alternating updates,
      A's growth amplifies the GCT gradient on B, which in turn is
      scaled by LoFT's inv(A@A.T + reg) — two opposing but unsynchronised
      effects that trigger a positive-feedback explosion (observed: loss
      500 000+ by epoch 3).  The reconstruction loss creates the same
      cross-coupling between A and B through per-node logits.

      Removing the auxiliary losses leaves a clean objective where LoFT's
      gradient rescaling acts only on the classification gradient, restoring
      the FFT-equivalent update trajectory that LoFT guarantees.

    Architecture:
      LoRA correction per layer — A=[r,d_in], B=[d_out,r] (LoFT convention)
      get_prompt: h @ A.T → GCN-aggregate → h_agg @ B.T

    Training:
      • Loss: cls_loss only
      • Optimizer: LoFTAdamW  (see downstream_task.py)
    """

    def __init__(
        self,
        dim_in_list: List[int],
        dim_out_list: List[int],
        r: int = 8,
        ct_weight: float = 0.5,
        rec_weight: float = 0.1,
    ):
        super().__init__()
        assert len(dim_in_list) == len(dim_out_list)
        self.num_layers = len(dim_in_list)
        self.r = r
        self.ct_weight  = ct_weight
        self.rec_weight = rec_weight

        # LoRA corrections  (LoFT convention: A=[r,d_in], B=[d_out,r])
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for d_in, d_out in zip(dim_in_list, dim_out_list):
            A = nn.Parameter(torch.empty(r, d_in))    # [r, d_in]
            B = nn.Parameter(torch.zeros(d_out, r))   # [d_out, r]
            nn.init.kaiming_normal_(A)
            self.lora_A.append(A)
            self.lora_B.append(B)

    def get_prompt(self, h: torch.Tensor, edge_index: torch.Tensor,
                   layer: int) -> torch.Tensor:
        """GCN-normalised LoRA correction (LoFT parameter convention).

        h_down = h @ A.T       [N, r]     (A.T transposes [r,d_in] → [d_in,r])
        h_agg  = D^{-1/2}-aggregate(h_down)
        p      = h_agg @ B.T   [N, d_out] (B.T transposes [d_out,r] → [r,d_out])
        """
        A = self.lora_A[layer]   # [r, d_in]
        B = self.lora_B[layer]   # [d_out, r]
        h_down = h @ A.T         # [N, r]

        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        row, col = edge_index_sl
        deg = degree(col, num_nodes=h.size(0), dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h_msg = h_down[col] * norm.unsqueeze(1)
        h_agg = torch.zeros_like(h_down)
        h_agg.scatter_add_(0, row.unsqueeze(1).expand_as(h_msg), h_msg)
        return h_agg @ B.T       # [N, d_out]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        dims = [(a.shape[1], b.shape[0]) for a, b in zip(self.lora_A, self.lora_B)]
        return (
            f"GraphLoFT(r={self.r}, layers={dims}, "
            f"ct={self.ct_weight}, rec={self.rec_weight}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# LoFTAdamW optimizer
# ---------------------------------------------------------------------------

class LoFTAdamW(torch.optim.Optimizer):
    """LoFT-aware AdamW for GNN LoRA fine-tuning.

    Faithful adaptation of LoFTAdamW (Tastan et al., ICLR 2026) for PyG GNNs.
    Works with nn.ParameterList naming (e.g. 'lora_A.0', 'lora_B.1').
    Non-LoRA parameters (classifier, projector) receive plain AdamW.

    Three LoFT mechanisms:
      1. Gradient rescaling:
             g̃_A = (B.T@B + εI)⁻¹ @ g_A    A=[r,d_in]
             g̃_B = g_B @ (A@A.T + εI)⁻¹    B=[d_out,r]
      2. Momentum reprojection through the other factor's change.
      3. Row-product second moments: tracks E[gᵢ ⊗ gᵢ] per row (shape [n,r,r]),
         reconstructs a full-space Adam denominator.

    Usage (downstream_task.py):
        wrapper = nn.ModuleDict({'classifier': clf, 'prompt': prompt})
        optimizer = LoFTAdamW(list(wrapper.parameters()), lr=lr,
                              weight_decay=wd, model=wrapper)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=1e-2,
                 model: nn.Module = None,
                 lora_A_name: str = 'lora_A',
                 lora_B_name: str = 'lora_B',
                 alternate_update: bool = True,
                 rescale_grads: bool = True,
                 reproject_momentum: bool = True,
                 reproject_second_moment: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lora_A_name = lora_A_name
        self.lora_B_name = lora_B_name
        self.alternate_update = alternate_update
        self.update_A = False   # start by updating B (B init=0)
        self.rescale_grads = rescale_grads
        self.reproject_momentum = reproject_momentum
        self.reproject_second_moment = reproject_second_moment
        self.lora_params_to_name: dict = {}
        self.lora_name_to_params: dict = {}
        self.old_params: dict = {}
        if model is not None:
            self._build_lora_name_maps(model)

    def _build_lora_name_maps(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.lora_A_name in name or self.lora_B_name in name:
                self.lora_params_to_name[param] = name
                self.lora_name_to_params[name]  = param

    def _other_name(self, name: str) -> str:
        if self.lora_A_name in name:
            return name.replace(self.lora_A_name, self.lora_B_name)
        return name.replace(self.lora_B_name, self.lora_A_name)

    def _do_update(self, p_name: Optional[str]) -> bool:
        if not self.alternate_update or p_name is None:
            return True
        if self.update_A and self.lora_B_name in p_name:
            return False
        if not self.update_A and self.lora_A_name in p_name:
            return False
        return True

    @staticmethod
    def _rescale(grad, other_p, is_A, eps_abs: float = 5e-3, eps_rel: float = 0.05):
        """Rescale gradient to remove LoRA parameterisation artifact.

        Returns (rescaled_grad, S), or (grad, None) when other_p is near-zero.

        When B ≈ 0 the LoRA output is identically zero and there is no
        parameterisation artifact to correct.  Skipping rescaling avoids
        dividing by a near-singular matrix (B.T @ B + ε I ≈ ε I → S ≈ I/ε
        which amplifies A's gradient by 1/ε and destabilises training).

        Adaptive epsilon: eps = eps_abs + eps_rel · mean(diag(M))
        ensures the condition number of (M + eps·I) is bounded regardless
        of M's scale, rather than using a fixed tiny constant.
        """
        if other_p.norm().item() < 1e-7:
            return grad, None  # B≈0: no artifact, skip to plain Adam

        if is_A:   # A=[r,d_in], other=B=[d_out,r] → M = B.T@B = [r,r]
            M = other_p.T @ other_p
            reg = eps_abs + eps_rel * M.diagonal().abs().mean().item()
            eye = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
            try:
                S = torch.linalg.inv(M + reg * eye)
            except torch.linalg.LinAlgError:
                return grad, None
            return S @ grad, S
        else:      # B=[d_out,r], other=A=[r,d_in] → M = A@A.T = [r,r]
            M = other_p @ other_p.T
            reg = eps_abs + eps_rel * M.diagonal().abs().mean().item()
            eye = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
            try:
                S = torch.linalg.inv(M + reg * eye)
            except torch.linalg.LinAlgError:
                return grad, None
            return grad @ S, S

    @staticmethod
    def _mom_sq(row_prod, T):
        """Diagonal of T.T @ P[i] @ T for each row i.  row_prod:(n,r,r) T:(r,d)"""
        temp = torch.matmul(row_prod, T)                    # (n,r,d)
        return (T.unsqueeze(0) * temp).sum(1).clamp(min=0)  # (n,d)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad  = p.grad.data
                state = self.state[p]
                p_name  = self.lora_params_to_name.get(p, None)
                is_lora = p_name is not None
                is_A    = is_lora and (self.lora_A_name in p_name)
                do_upd  = self._do_update(p_name)

                # ---- initialise state ----------------------------------------
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Always keep exp_avg_sq for a safe AdamW fallback when LoRA
                    # gradient rescaling is skipped (scaling is None).
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.reproject_second_moment and is_lora:
                        if is_A:
                            r, n = p.shape   # A=[r, d_in=n]
                            state['row_products'] = torch.zeros(
                                n, r, r, device=p.device, dtype=p.dtype)
                        else:
                            m, r = p.shape   # B=[d_out=m, r]
                            state['row_products'] = torch.zeros(
                                m, r, r, device=p.device, dtype=p.dtype)

                # ---- gradient rescaling + momentum reprojection ---------------
                scaling = None
                if self.rescale_grads and is_lora:
                    other_p = self.lora_name_to_params.get(self._other_name(p_name))
                    if other_p is not None:
                        grad, scaling = self._rescale(grad, other_p, is_A)
                        # Only reproject when rescaling was actually applied
                        # (scaling is None when B≈0 and we fell back to plain Adam)
                        if scaling is not None and self.reproject_momentum:
                            old = self.old_params.get(self._other_name(p_name))
                            if old is not None:
                                R = (scaling @ (other_p.T @ old) if is_A
                                     else (old @ other_p.T) @ scaling)
                                m = state['exp_avg']
                                state['exp_avg'] = R @ m if is_A else m @ R
                                if self.reproject_second_moment and is_lora:
                                    rp = state['row_products']
                                    state['row_products'] = (
                                        R.T[None] @ rp @ R[None] if is_A
                                        else R[None] @ rp @ R.T[None])
                            self.old_params[self._other_name(p_name)] = \
                                other_p.data.detach().clone()

                beta1, beta2 = group['betas']
                state['step'] += 1
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)

                state['exp_avg_sq'].mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)
                if self.reproject_second_moment and is_lora:
                    vec = grad.T if is_A else grad    # [n,r] or [m,r]
                    state['row_products'].mul_(beta2).baddbmm_(
                        vec[:, :, None], vec[:, None, :], alpha=1 - beta2)

                if not do_upd:
                    continue

                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['weight_decay'])

                if self.reproject_second_moment and is_lora and scaling is not None:
                    other_p = self.lora_name_to_params.get(self._other_name(p_name))
                    if other_p is not None:
                        m   = state['exp_avg']
                        rp  = state['row_products']
                        T   = other_p.T if is_A else other_p
                        fu  = other_p @ m if is_A else m @ other_p  # full-space update
                        msq = self._mom_sq(rp, T)
                        denom = (msq.sqrt() / math.sqrt(bc2)).add_(group['eps'])
                        fu.div_(denom.T if is_A else denom)
                        pu = (scaling @ (other_p.T @ fu) if is_A
                              else (fu @ other_p.T) @ scaling)
                        p.data.add_(pu, alpha=-group['lr'] / bc1)
                        continue

                denom = (state['exp_avg_sq'].sqrt() /
                         math.sqrt(bc2)).add_(group['eps'])
                p.data.addcdiv_(state['exp_avg'], denom,
                                value=-group['lr'] / bc1)

        if self.alternate_update:
            self.update_A = not self.update_A
        return loss
