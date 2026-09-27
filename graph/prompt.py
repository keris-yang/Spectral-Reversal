"""
Prompt module for graph-level GNN (GIN).

Classes:
    - SRP: Spectral Reverse Prompt for GIN (graph classification)

Key difference from the node (GCN) version:
    GIN applies BatchNorm before the prompt addition. To handle degree
    heterogeneity across graphs, each layer's prompt is additionally scaled
    by a learnable degree gate:
        p_raw = LayerNorm(z @ Q + b)
        p     = p_raw * sigmoid(log(deg + 1) * w_d + b_d)

Decomp decision per layer:
    use_decomp[i] = True  iff  d_in > k + null_pca_dim

For typical GIN (d_in < hidden_dim at layer 0, e.g. ENZYMES: 18 < 128):
    All layers use the projective fallback (no algebraic null space).
For datasets with large input features (d_in > hidden_dim + null_pca_dim):
    Layer 0 activates the dual-channel path.
"""

import torch
import torch.nn as nn
from typing import List
from torch_geometric.utils import degree


def _estimate_rank(W_cpu: torch.Tensor, tol_ratio: float = 1e-3) -> int:
    """Numerically estimate rank(W) via SVD."""
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(W_cpu, full_matrices=False)
    threshold = tol_ratio * S[0].item()
    return int((S > threshold).sum().item())


class SRP(nn.Module):
    r"""SRP — Spectral Reverse Prompt for GIN (graph classification).

    Decomposes each frozen GNN weight W = U Σ Vᵀ via SVD, then applies a
    dual-channel prompt that inverts the spectral priority of pre-training:

      Spectral channel: learnable soft-threshold mask amplifies weak (small-σ)
        directions and suppresses dominant (large-σ) directions.
            soft_mask_j = sigmoid((τ · σ_max − σ_j) · 10)
            z_spec = (h @ V  ⊙  soft_mask) @ A

      Null-space channel (high-dim input, use_decomp=True):
            z_null = (h @ N_mat) @ B

      Projective fallback (low-dim input, use_decomp=False):
            h_weak = h − (h @ V  ⊙  (1 − soft_mask)) @ Vᵀ
            z = h_weak @ P

    GIN-specific output (every layer):
        p_raw = LayerNorm(z @ Q + b)
        p     = p_raw * sigmoid(log(deg + 1) * w_d + b_d)

    Decomp decision:
        use_decomp[i] = (d_in > k + null_pca_dim)
    """

    def __init__(
        self,
        dim_in_list: List[int],
        dim_out_list: List[int],
        weight_matrices: List[torch.Tensor],
        node_features: torch.Tensor,
        null_pca_dim: int = 16,
        r_shared: int = 32,
    ):
        super().__init__()
        self.num_layers = len(dim_in_list)
        self.null_pca_dim = null_pca_dim
        self.use_decomp = []

        self.threshold  = nn.ParameterList()
        self.proj_Q     = nn.ParameterList()
        self.proj_b     = nn.ParameterList()
        self.deg_w      = nn.ParameterList()
        self.deg_b      = nn.ParameterList()
        self.prompt_ln  = nn.ModuleList()

        for i, (W, d_in, d_out) in enumerate(
            zip(weight_matrices, dim_in_list, dim_out_list)
        ):
            W_cpu = W.detach().cpu()
            with torch.no_grad():
                _U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
                V = Vh.T      # [d_in, k]
                k = V.shape[1]

            self.register_buffer(f'S_{i}', S.clone())
            self.register_buffer(f'V_{i}', V.clone())
            self.threshold.append(nn.Parameter(torch.tensor(0.5)))

            rs = min(r_shared, d_out)

            need_decomp = d_in > (k + null_pca_dim)
            self.use_decomp.append(need_decomp)

            if need_decomp:
                X = node_features.float().cpu()
                X_null = X - X @ V @ V.T
                X_null_c = X_null - X_null.mean(dim=0, keepdim=True)
                _, _, Vn = torch.linalg.svd(X_null_c, full_matrices=False)
                m = min(null_pca_dim, Vn.shape[0])
                N_mat = Vn[:m].T           # [d_in, m]
                self.register_buffer(f'N_{i}', N_mat.clone())
                setattr(self, f'spec_A_{i}', nn.Parameter(
                    torch.randn(k, rs) * (2.0 / (k + rs)) ** 0.5
                ))
                setattr(self, f'null_B_{i}', nn.Parameter(
                    torch.randn(m, rs) * (2.0 / (m + rs)) ** 0.5
                ))
            else:
                self.register_buffer(f'N_{i}', None)
                setattr(self, f'proj_P_{i}', nn.Parameter(
                    torch.randn(d_in, rs) * (2.0 / (d_in + rs)) ** 0.5
                ))

            self.proj_Q.append(nn.Parameter(
                torch.randn(rs, d_out) * (2.0 / (rs + d_out)) ** 0.5
            ))
            self.proj_b.append(nn.Parameter(torch.zeros(d_out)))
            self.deg_w.append(nn.Parameter(torch.tensor(0.0)))
            self.deg_b.append(nn.Parameter(torch.tensor(0.0)))
            self.prompt_ln.append(nn.LayerNorm(d_out, elementwise_affine=False))

    def get_prompt(self, h, edge_index, layer):
        V = getattr(self, f'V_{layer}')
        S = getattr(self, f'S_{layer}')

        tau = torch.sigmoid(self.threshold[layer])
        threshold_val = tau * S.max()
        soft_mask = torch.sigmoid((threshold_val - S) * 10.0)

        if self.use_decomp[layer]:
            N_mat  = getattr(self, f'N_{layer}')
            spec_A = getattr(self, f'spec_A_{layer}')
            null_B = getattr(self, f'null_B_{layer}')
            h_v    = h @ V
            h_null = h @ N_mat
            z_spec = (h_v * soft_mask.unsqueeze(0)) @ spec_A
            z_null = h_null @ null_B
            z = z_spec + z_null
        else:
            h_v    = h @ V
            h_weak = h - (h_v * (1 - soft_mask).unsqueeze(0)) @ V.T
            proj_P = getattr(self, f'proj_P_{layer}')
            z = h_weak @ proj_P

        p = self.prompt_ln[layer](z @ self.proj_Q[layer] + self.proj_b[layer])

        row, _ = edge_index
        deg = degree(row, num_nodes=h.size(0), dtype=h.dtype)
        deg_scale = torch.sigmoid(
            torch.log(deg + 1.0) * self.deg_w[layer] + self.deg_b[layer]
        ).unsqueeze(1)
        return p * deg_scale

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def decomp_summary(self) -> str:
        lines = []
        for i, flag in enumerate(self.use_decomp):
            mode = 'dual-channel (spectral + null-space)' if flag else 'projective fallback (no null-space)'
            lines.append(f"  Layer {i}: {mode}")
        return "\n".join(lines)
