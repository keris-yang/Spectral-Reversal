"""
Prompt module for node-level GNN (GCN).

Classes:
    - SRP: Spectral Reverse Prompt (static dimension-based decomp decision)

Decomp decision per layer:
    use_decomp[i] = True  iff  d_in > k + null_pca_dim
      (dual-channel: spectral reversal + null-space PCA)
    else:
      projective fallback (single spectral-filtered channel)

For Cora (1433 → 128 → 128):
    with the released default m=16:
    Layer 0: d_in=1433, k=128, so d_in > k + m → dual-channel
    Layer 1: d_in=128 ≤ k + m → projective fallback
"""

import torch
import torch.nn as nn
from typing import List


def _estimate_rank(W_cpu: torch.Tensor, tol_ratio: float = 1e-3) -> int:
    """Numerically estimate rank(W) via SVD.
    Directions with singular value < tol_ratio * sigma_max are considered null.
    """
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(W_cpu, full_matrices=False)
    threshold = tol_ratio * S[0].item()
    return int((S > threshold).sum().item())


class SRP(nn.Module):
    r"""SRP — Spectral Reverse Prompt for GCN (node classification).

    Decomposes each frozen GNN weight W = U Σ Vᵀ via SVD, then applies a
    dual-channel prompt that *inverts* the spectral priority of pre-training:

      Spectral channel: projects h onto right singular vectors V, applies a
        learnable soft-threshold mask that amplifies weak (small-σ) directions
        and suppresses dominant (large-σ) directions.
            soft_mask_j = sigmoid((τ · σ_max − σ_j) · 10)
            z_spec = (h @ V  ⊙  soft_mask) @ A

      Null-space channel (high-dim input only, use_decomp=True):
        performs PCA within the algebraic null space ker(W) of the data
        to extract discriminative signal completely invisible to W.
            z_null = (h @ N_mat) @ B,  where N_mat spans ker(W) directions

      Projective fallback (low-dim input, use_decomp=False):
        constructs a spectrally-filtered h_weak (strong directions subtracted)
        and projects it via a learnable matrix P.

    Final prompt: p = z @ Q + b, added post-convolution to the hidden state.

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

        self.threshold = nn.ParameterList()
        self.proj_Q = nn.ParameterList()
        self.proj_b = nn.ParameterList()

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

    def get_prompt(self, h, edge_index, layer):
        V = getattr(self, f'V_{layer}')
        S = getattr(self, f'S_{layer}')

        tau = torch.sigmoid(self.threshold[layer])
        threshold_val = tau * S.max()
        soft_mask = torch.sigmoid((threshold_val - S) * 10.0)

        if self.use_decomp[layer]:
            N_mat = getattr(self, f'N_{layer}')
            spec_A = getattr(self, f'spec_A_{layer}')
            null_B = getattr(self, f'null_B_{layer}')
            h_v = h @ V
            h_null = h @ N_mat
            z_spec = (h_v * soft_mask.unsqueeze(0)) @ spec_A
            z_null = h_null @ null_B
            z = z_spec + z_null
        else:
            h_v = h @ V
            h_weak = h - (h_v * (1 - soft_mask).unsqueeze(0)) @ V.T
            proj_P = getattr(self, f'proj_P_{layer}')
            z = h_weak @ proj_P

        return z @ self.proj_Q[layer] + self.proj_b[layer]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def decomp_summary(self) -> str:
        lines = []
        for i, flag in enumerate(self.use_decomp):
            mode = 'dual-channel (spectral + null-space)' if flag else 'projective fallback (no null-space)'
            lines.append(f"  Layer {i}: {mode}")
        return "\n".join(lines)
