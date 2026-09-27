import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_add_pool, MessagePassing, inits
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = Linear(input_dim, output_dim, bias=False, weight_initializer='glorot')
        self.bias = nn.Parameter(torch.empty(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        inits.zeros(self.bias)

    def forward(self, x, edge_index, edge_prompt=False):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm, edge_attr=edge_prompt)

        out = out + self.bias
        return out

    def message(self, x_j, norm, edge_attr):
        if edge_attr is not False:
            return norm.view(-1, 1) * self.lin(x_j + edge_attr)
        else:
            return norm.view(-1, 1) * self.lin(x_j)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_ratio=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.drop_ratio = drop_ratio

    def forward(self, data, prompt_type=None, prompt=None, pooling=False,
                return_extra=False):
        """Forward pass.

        Args:
            data:        PyG Data (or Batch) object.
            prompt_type: string prompt identifier or None.
            prompt:      prompt module instance or None.
            pooling:     'mean' | 'target' | False.
            return_extra: if True and prompt_type=='GraphLoRA', also return the
                          pooled frozen-branch and LoRA-branch embeddings as a
                          3-tuple (emb, emb_frozen, emb_lora) for the GCT
                          contrastive loss.  Ignored for all other prompt types.
        """
        assert pooling in ['mean', 'target', False]
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ------------------------------------------------------------------ #
        # GraphLoRA: projector + frozen/LoRA branches at last layer for GCT #
        # ------------------------------------------------------------------ #
        if prompt_type == 'GraphLoRA':
            # Layer 0: frozen + LoRA + ReLU
            h = x
            z = self.conv1(h, edge_index, edge_prompt=False)
            p = prompt.get_prompt(h, edge_index, layer=0)
            x = F.relu(z + p)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)

            # Layer 1 (last): keep frozen and LoRA branches separate for GCT
            h = x
            z_frozen = self.conv2(h, edge_index, edge_prompt=False)
            z_lora   = prompt.get_prompt(h, edge_index, layer=1)
            x_total  = z_frozen + z_lora

            if pooling == 'mean':
                emb = global_mean_pool(x_total, batch.long())
                if return_extra:
                    emb_frozen = global_mean_pool(z_frozen, batch.long())
                    emb_lora   = global_mean_pool(z_lora,   batch.long())
                    return emb, emb_frozen, emb_lora
                return emb
            if pooling == 'target':
                idx = data.ptr[:-1] + data.target_node
                emb = x_total[idx]
                if return_extra:
                    return emb, z_frozen[idx], z_lora[idx]
                return emb
            if return_extra:
                return x_total, z_frozen, z_lora
            return x_total

        # ------------------------------------------------------------------ #
        # GraphLoFT: projector + LoRA correction, cls_loss only.             #
        # GCT / rec auxiliary losses are intentionally omitted — they create  #
        # A↔B gradient cross-coupling that LoFT's rescaling amplifies into   #
        # instability. See graphlora.py GraphLoFT docstring for details.      #
        # ------------------------------------------------------------------ #
        elif prompt_type == 'GraphLoFT':
            h = x
            z = self.conv1(h, edge_index, edge_prompt=False)
            p = prompt.get_prompt(h, edge_index, layer=0)
            x = F.relu(z + p)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)

            h = x
            z = self.conv2(h, edge_index, edge_prompt=False)
            p = prompt.get_prompt(h, edge_index, layer=1)
            x = z + p

        # ------------------------------------------------------------------ #
        # SRP variants: standard post-conv injection                         #
        # ------------------------------------------------------------------ #
        elif prompt_type is not None and prompt_type.startswith('SRP'):
            # Layer 0
            h = x
            z = self.conv1(h, edge_index, edge_prompt=False)
            p = prompt.get_prompt(h, edge_index, layer=0)
            x = F.relu(z + p)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)

            # Layer 1
            h = x
            z = self.conv2(h, edge_index, edge_prompt=False)
            p = prompt.get_prompt(h, edge_index, layer=1)
            x = z + p

        else:
            # === Pre-W Prompts: EdgePrompt ===
            # Layer 0
            h = x
            edge_prompt = False
            if prompt_type == 'EdgePrompt':
                edge_prompt = prompt.get_prompt(h, edge_index, layer=0)

            x = self.conv1(h, edge_index, edge_prompt)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)

            # Layer 1
            h = x
            edge_prompt = False
            if prompt_type == 'EdgePrompt':
                edge_prompt = prompt.get_prompt(h, edge_index, layer=1)

            x = self.conv2(h, edge_index, edge_prompt)

        if pooling == 'mean':
            graph_emb = global_mean_pool(x, batch.long())
            return graph_emb
        if pooling == 'target':
            graph_emb = x[data.ptr[:-1] + data.target_node]
            return graph_emb

        return x

    @torch.no_grad()
    def get_layer_inputs(self, data):
        """Collect h_i (input to layer i) for all layers, without prompt.

        Returns [h_0, h_1]:
          h_0 : raw node features        (shape [N, input_dim])
          h_1 : output of conv1 + ReLU   (shape [N, hidden_dim])

        Used for data-driven null-space detection: Layer 1 (128×128) may have
        genuine null space after contrastive pretraining (low-rank W).
        """
        was_training = self.training
        self.eval()
        x, edge_index = data.x, data.edge_index
        h0 = x
        z1 = self.conv1(h0, edge_index, edge_prompt=False)
        h1 = F.dropout(F.relu(z1), p=self.drop_ratio, training=False)
        if was_training:
            self.train()
        return [h0, h1]
