"""
Ablation variants of SRP (Spectral Reverse Prompt) for node-level GNN (GCN).

All classes share the same constructor signature as SRP and produce prompts of
shape [N, d_out], added post-convolution to the hidden state.

Classes:
    SRP_NM  — SRP without null-space channel (spectral channel only for high-dim)
    SRP_NS  — SRP without spectral channel   (null-space channel only for high-dim)
    SRP_NR  — SRP without Reverse mechanism  (no soft-threshold mask; uniform mask=1)
    SRP_Bi  — SRP with bidirectional mask    (U-shaped: amplifies weak AND strong)
"""

import torch
import torch.nn as nn
from typing import List


# =============================================================================
# SRP_NM: SRP without null-space channel
# =============================================================================

class SRP_NM(nn.Module):
    r"""SRP_NM — SRP without null-space channel.

    Ablation: the null-space PCA channel (Channel B) is removed.
    Only the spectral reversal channel remains.

    High-dim layers (use_decomp=True):
        z = (h @ V  ⊙  soft_mask) @ spec_A      [spectral channel only]

    Low-dim layers (use_decomp=False):
        z = h_weak @ proj_P                       [same projective fallback as SRP]
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
                setattr(self, f'spec_A_{i}', nn.Parameter(
                    torch.randn(k, rs) * (2.0 / (k + rs)) ** 0.5
                ))
            else:
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
            spec_A = getattr(self, f'spec_A_{layer}')
            h_v = h @ V
            z = (h_v * soft_mask.unsqueeze(0)) @ spec_A
        else:
            h_v = h @ V
            h_weak = h - (h_v * (1 - soft_mask).unsqueeze(0)) @ V.T
            proj_P = getattr(self, f'proj_P_{layer}')
            z = h_weak @ proj_P

        return z @ self.proj_Q[layer] + self.proj_b[layer]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# SRP_NS: SRP without spectral channel
# =============================================================================

class SRP_NS(nn.Module):
    r"""SRP_NS — SRP without spectral channel.

    Ablation: the spectral reversal channel (Channel A, with soft-threshold mask)
    is removed. Only the null-space PCA channel remains.

    High-dim layers (use_decomp=True):
        z = (h @ N_mat) @ null_B                  [null-space channel only]

    Low-dim layers (use_decomp=False):
        No algebraic null space exists; falls back to a plain linear projection
        of h without any spectral filtering.
        z = h @ proj_P
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
        self.use_decomp = []

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
        if self.use_decomp[layer]:
            N_mat = getattr(self, f'N_{layer}')
            null_B = getattr(self, f'null_B_{layer}')
            z = (h @ N_mat) @ null_B
        else:
            proj_P = getattr(self, f'proj_P_{layer}')
            z = h @ proj_P

        return z @ self.proj_Q[layer] + self.proj_b[layer]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# SRP_NR: SRP without Reverse mechanism (no soft-threshold mask)
# =============================================================================

class SRP_NR(nn.Module):
    r"""SRP_NR — SRP without the Reverse mechanism.

    Ablation: the learnable soft-threshold mask (the spectral reversal) is removed.
    All singular directions receive uniform weight 1, so no reversal of spectral
    priority occurs. The null-space channel is still active for high-dim layers.

    High-dim layers (use_decomp=True):
        z = h_v @ spec_A + h_null @ null_B        [no mask weighting]

    Low-dim layers (use_decomp=False):
        Without the mask, h_weak = h (strong directions are NOT subtracted).
        z = h @ proj_P
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
        self.use_decomp = []

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

            self.register_buffer(f'V_{i}', V.clone())

            rs = min(r_shared, d_out)
            need_decomp = d_in > (k + null_pca_dim)
            self.use_decomp.append(need_decomp)

            if need_decomp:
                X = node_features.float().cpu()
                X_null = X - X @ V @ V.T
                X_null_c = X_null - X_null.mean(dim=0, keepdim=True)
                _, _, Vn = torch.linalg.svd(X_null_c, full_matrices=False)
                m = min(null_pca_dim, Vn.shape[0])
                N_mat = Vn[:m].T
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

        if self.use_decomp[layer]:
            N_mat = getattr(self, f'N_{layer}')
            spec_A = getattr(self, f'spec_A_{layer}')
            null_B = getattr(self, f'null_B_{layer}')
            h_v = h @ V
            h_null = h @ N_mat
            z = h_v @ spec_A + h_null @ null_B    # no mask
        else:
            proj_P = getattr(self, f'proj_P_{layer}')
            z = h @ proj_P    # no spectral filtering (mask=1 ⟹ h_weak = h)

        return z @ self.proj_Q[layer] + self.proj_b[layer]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# SRP_Bi: SRP with bidirectional (U-shaped) mask
# =============================================================================

class SRP_Bi(nn.Module):
    r"""SRP_Bi — SRP with bidirectional spectral mask.

    Ablation: the spectral channel uses a U-shaped mask that amplifies BOTH
    weak (small-σ) AND strong (large-σ) directions, suppressing only the middle
    range. This contrasts with SRP's one-sided mask that amplifies weak directions
    only (spectral reversal).

    U-shaped mask per layer (two learnable thresholds τ_low, τ_high):
        mask_j = sigmoid((τ_low · σ_max − σ_j) · 10)   ← amplify weak end
               + sigmoid((σ_j − τ_high · σ_max) · 10)  ← preserve strong end
        clamped to [0, 1]

    High-dim layers (use_decomp=True):
        z = (h @ V  ⊙  u_mask) @ spec_A + (h @ N_mat) @ null_B

    Low-dim layers (use_decomp=False):
        h_weak = h − (h @ V  ⊙  (1 − u_mask)) @ Vᵀ   (non-reversed suppression)
        z = h_weak @ proj_P
    """

    def __init__(
        self,
        dim_in_list: List[int],
        dim_out_list: List[int],
        weight_matrices: List[torch.Tensor],
        node_features: torch.Tensor,
        null_pca_dim: int = 16,
        r_shared: int = 32,
        tau_low_init: float = 0.3,
        tau_high_init: float = 0.7,
    ):
        super().__init__()
        self.num_layers = len(dim_in_list)
        self.use_decomp = []

        self.tau_low = nn.ParameterList()
        self.tau_high = nn.ParameterList()
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
            self.tau_low.append(nn.Parameter(torch.tensor(tau_low_init)))
            self.tau_high.append(nn.Parameter(torch.tensor(tau_high_init)))

            rs = min(r_shared, d_out)
            need_decomp = d_in > (k + null_pca_dim)
            self.use_decomp.append(need_decomp)

            if need_decomp:
                X = node_features.float().cpu()
                X_null = X - X @ V @ V.T
                X_null_c = X_null - X_null.mean(dim=0, keepdim=True)
                _, _, Vn = torch.linalg.svd(X_null_c, full_matrices=False)
                m = min(null_pca_dim, Vn.shape[0])
                N_mat = Vn[:m].T
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

        tl = torch.sigmoid(self.tau_low[layer])
        th = torch.sigmoid(self.tau_high[layer])
        S_max = S.max()
        u_mask = (
            torch.sigmoid((tl * S_max - S) * 10.0)    # weak end ≈ 1
          + torch.sigmoid((S - th * S_max) * 10.0)    # strong end ≈ 1
        ).clamp(max=1.0)

        if self.use_decomp[layer]:
            N_mat = getattr(self, f'N_{layer}')
            spec_A = getattr(self, f'spec_A_{layer}')
            null_B = getattr(self, f'null_B_{layer}')
            h_v = h @ V
            h_null = h @ N_mat
            z_spec = (h_v * u_mask.unsqueeze(0)) @ spec_A
            z_null = h_null @ null_B
            z = z_spec + z_null
        else:
            h_v = h @ V
            h_weak = h - (h_v * (1 - u_mask).unsqueeze(0)) @ V.T
            proj_P = getattr(self, f'proj_P_{layer}')
            z = h_weak @ proj_P

        return z @ self.proj_Q[layer] + self.proj_b[layer]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
