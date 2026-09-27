"""
GraphLoRA and GraphLoFT adapters for graph-level GNN (GIN backbone).

GraphLoRA (Wei et al., 2024) — faithful adaptation to the EdgePrompt framework.

    Two modules (node feature projection step removed):
    1. Structural Knowledge Transfer — separate frozen/LoRA branches at the
       last GIN layer; GCT contrastive loss between the two branches
    2. Structure-aware Regularization — per-graph adjacency reconstruction loss

    Combined training loss:
        L = cls_loss + ct_weight · L_ct + rec_weight · L_rec

GraphLoFT (proposed) — GraphLoRA architecture + LoFT training dynamics:

    Extends GraphLoRA with LoFT (Tastan et al., ICLR 2026).
    Node feature projection step removed; parameters use LoFT's convention
    (A=[r,d_in], B=[d_out,r]) so LoFTAdamW's gradient rescaling and
    momentum reprojection apply.  Trained with LoFTAdamW.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch_geometric.utils import degree, to_dense_adj


# ---------------------------------------------------------------------------
# GraphLoRA  (GIN / graph classification)
# ---------------------------------------------------------------------------

class GraphLoRA(nn.Module):
    """Full GraphLoRA for a frozen multi-layer GIN (graph classification).

    Two modules (LoRA knowledge transfer + structure-aware reg).
    Node feature projector removed: not applicable to same-graph setting.
    Parameters: lora_A=[d_in, r], lora_B=[r, d_out]  (GraphLoRA convention).
    Trained with standard Adam.
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

        self.lora_A    = nn.ParameterList()
        self.lora_B    = nn.ParameterList()
        self.eps       = nn.ParameterList()
        self.deg_w     = nn.ParameterList()
        self.deg_b     = nn.ParameterList()
        self.prompt_ln = nn.ModuleList()

        for d_in, d_out in zip(dim_in_list, dim_out_list):
            A = nn.Parameter(torch.empty(d_in, r))
            B = nn.Parameter(torch.zeros(r, d_out))
            nn.init.kaiming_normal_(A)
            self.lora_A.append(A)
            self.lora_B.append(B)
            self.eps.append(nn.Parameter(torch.tensor(0.0)))
            self.deg_w.append(nn.Parameter(torch.tensor(0.0)))
            self.deg_b.append(nn.Parameter(torch.tensor(0.0)))
            self.prompt_ln.append(nn.LayerNorm(d_out, elementwise_affine=False))

    def get_prompt(self, h: torch.Tensor, edge_index: torch.Tensor,
                   layer: int) -> torch.Tensor:
        """GIN-aggregated LoRA correction.

        h_center = (1+ε)·h + Σ relu(h[j])
        z        = h_center @ A @ B          [N, d_out]
        p        = LayerNorm(z) · deg_gate
        """
        A   = self.lora_A[layer]   # [d_in, r]
        B   = self.lora_B[layer]   # [r, d_out]
        eps = self.eps[layer]
        row, col = edge_index
        N = h.size(0)

        h_msg = F.relu(h[col])
        h_agg = torch.zeros_like(h)
        h_agg.scatter_add_(0, row.unsqueeze(1).expand_as(h_msg), h_msg)
        h_center = (1.0 + eps) * h + h_agg
        z = h_center @ A @ B

        p = self.prompt_ln[layer](z)
        deg = degree(row, num_nodes=N, dtype=h.dtype)
        deg_gate = torch.sigmoid(
            torch.log(deg + 1.0) * self.deg_w[layer] + self.deg_b[layer]
        ).unsqueeze(1)
        return p * deg_gate

    # ------ shared static losses ------

    @staticmethod
    def gct_loss(
        z_frozen: torch.Tensor,
        z_lora: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        tau: float = 0.5,
        sup_weight: float = 0.2,
    ) -> torch.Tensor:
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

        def _one(za, zb):
            refl  = torch.exp(za @ za.T / tau)
            cross = torch.exp(za @ zb.T / tau)
            numer = (mask * cross).sum(1)
            denom = refl.sum(1) + cross.sum(1) - refl.diag()
            return -torch.log(numer / denom.clamp(min=1e-10)).mean()

        return 0.5 * (_one(z1, z2) + _one(z2, z1))

    @staticmethod
    def reconstruction_loss(
        logits_per_node: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if num_nodes > 500:
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
# GraphLoFT  (GIN / graph classification)
# ---------------------------------------------------------------------------

class GraphLoFT(nn.Module):
    """GraphLoFT — LoFT-style LoRA correction (GIN).

    Node feature projector removed: not applicable to same-graph setting.
    Omits GCT contrastive loss and adjacency reconstruction loss.
    See node/graphlora.py GraphLoFT docstring for full rationale.

    Training: cls_loss only, optimised with LoFTAdamW.
    Parameters: A=[r,d_in], B=[d_out,r] (LoFT convention).
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

        # LoFT convention: A=[r, d_in], B=[d_out, r]
        self.lora_A    = nn.ParameterList()
        self.lora_B    = nn.ParameterList()
        self.eps       = nn.ParameterList()
        self.deg_w     = nn.ParameterList()
        self.deg_b     = nn.ParameterList()
        self.prompt_ln = nn.ModuleList()

        for d_in, d_out in zip(dim_in_list, dim_out_list):
            A = nn.Parameter(torch.empty(r, d_in))    # [r, d_in]
            B = nn.Parameter(torch.zeros(d_out, r))   # [d_out, r]
            nn.init.kaiming_normal_(A)
            self.lora_A.append(A)
            self.lora_B.append(B)
            self.eps.append(nn.Parameter(torch.tensor(0.0)))
            self.deg_w.append(nn.Parameter(torch.tensor(0.0)))
            self.deg_b.append(nn.Parameter(torch.tensor(0.0)))
            self.prompt_ln.append(nn.LayerNorm(d_out, elementwise_affine=False))

    def get_prompt(self, h: torch.Tensor, edge_index: torch.Tensor,
                   layer: int) -> torch.Tensor:
        """GIN-aggregated LoRA correction (LoFT parameter convention).

        h_center = (1+ε)·h + Σ relu(h[j])
        h_down   = h_center @ A.T    [N, r]
        z        = h_down   @ B.T    [N, d_out]
        p        = LayerNorm(z) · deg_gate
        """
        A   = self.lora_A[layer]   # [r, d_in]
        B   = self.lora_B[layer]   # [d_out, r]
        eps = self.eps[layer]
        row, col = edge_index
        N = h.size(0)

        h_msg = F.relu(h[col])
        h_agg = torch.zeros_like(h)
        h_agg.scatter_add_(0, row.unsqueeze(1).expand_as(h_msg), h_msg)
        h_center = (1.0 + eps) * h + h_agg
        h_down   = h_center @ A.T    # [N, r]
        z        = h_down   @ B.T    # [N, d_out]

        p = self.prompt_ln[layer](z)
        deg = degree(row, num_nodes=N, dtype=h.dtype)
        deg_gate = torch.sigmoid(
            torch.log(deg + 1.0) * self.deg_w[layer] + self.deg_b[layer]
        ).unsqueeze(1)
        return p * deg_gate

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
# LoFTAdamW optimizer (shared by GraphLoFT node and graph)
# ---------------------------------------------------------------------------

class LoFTAdamW(torch.optim.Optimizer):
    """LoFT-aware AdamW for GNN LoRA fine-tuning.

    Faithful adaptation of LoFTAdamW (Tastan et al., ICLR 2026) for PyG GNNs.
    Non-LoRA parameters (classifier, projector) receive plain AdamW.

    Usage:
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
        self.update_A = False
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
        See node/graphlora.py for full explanation.
        """
        if other_p.norm().item() < 1e-7:
            return grad, None

        if is_A:
            M = other_p.T @ other_p
            reg = eps_abs + eps_rel * M.diagonal().abs().mean().item()
            eye = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
            try:
                S = torch.linalg.inv(M + reg * eye)
            except torch.linalg.LinAlgError:
                return grad, None
            return S @ grad, S
        else:
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
        temp = torch.matmul(row_prod, T)
        return (T.unsqueeze(0) * temp).sum(1).clamp(min=0)

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

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.reproject_second_moment and is_lora:
                        if is_A:
                            r, n = p.shape
                            state['row_products'] = torch.zeros(
                                n, r, r, device=p.device, dtype=p.dtype)
                        else:
                            m, r = p.shape
                            state['row_products'] = torch.zeros(
                                m, r, r, device=p.device, dtype=p.dtype)

                scaling = None
                if self.rescale_grads and is_lora:
                    other_p = self.lora_name_to_params.get(self._other_name(p_name))
                    if other_p is not None:
                        grad, scaling = self._rescale(grad, other_p, is_A)
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
                    vec = grad.T if is_A else grad
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
                        fu  = other_p @ m if is_A else m @ other_p
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
