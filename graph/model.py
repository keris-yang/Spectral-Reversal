import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool


class GINConv(MessagePassing):
    def __init__(self, input_dim, hidden_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_prompt=False):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_prompt))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not False:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


class GIN(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim, drop_ratio=0.5):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GINConv(input_dim=input_dim, hidden_dim=hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for layer in range(num_layer - 1):
            self.convs.append(GINConv(input_dim=hidden_dim, hidden_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data, prompt_type=None, prompt=False, pooling=False):
        """Forward pass.

        Args:
            data:        PyG Data (or Batch) object.
            prompt_type: string prompt identifier or None.
            prompt:      prompt module instance or None.
            pooling:     'mean' | False.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ------------------------------------------------------------------ #
        # SRP variants: standard post-BN injection                          #
        # ------------------------------------------------------------------ #
        h_list = [x]

        for layer in range(self.num_layer):
            h = h_list[layer]

            if prompt_type is not None and prompt_type.startswith('SRP'):
                x = self.convs[layer](h, edge_index, edge_prompt=False)
                x = self.batch_norms[layer](x)
                p = prompt.get_prompt(h, edge_index, layer=layer)
                x = x + p

            else:
                # === Pre-W Prompts: EdgePrompt ===
                edge_prompt = False
                if prompt_type == 'EdgePrompt':
                    edge_prompt = prompt.get_prompt(h, edge_index, layer=layer)
                x = self.convs[layer](h, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

            if layer == self.num_layer - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            h_list.append(x)

        node_emb = h_list[-1]
        if pooling == 'mean':
            graph_emb = global_mean_pool(node_emb, batch)
            return graph_emb

        return node_emb

    @torch.no_grad()
    def get_layer_inputs(self, data):
        """Collect h_i (input to layer i) for all layers, without prompt.

        Returns a list of tensors [h_0, h_1, ..., h_{L-1}]:
          h_0 : raw node features  (shape [N_total, input_dim])
          h_i : output of layer i-1 after BN+ReLU+Dropout (shape [N_total, hidden_dim])

        Used for NS-PCA so every layer can compute its null-space from the actual
        data distribution at that layer's input space.
        """
        was_training = self.training
        self.eval()
        x, edge_index = data.x, data.edge_index
        h_list = [x]
        for layer in range(self.num_layer):
            h = h_list[layer]
            x = self.convs[layer](h, edge_index, edge_prompt=False)
            x = self.batch_norms[layer](x)
            if layer < self.num_layer - 1:
                x = F.dropout(F.relu(x), self.drop_ratio, training=False)
            else:
                x = F.dropout(x, self.drop_ratio, training=False)
            h_list.append(x)
        if was_training:
            self.train()
        return h_list[:self.num_layer]  # [h_0, ..., h_{L-1}]
