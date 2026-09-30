"""
Downstream task: Graph classification with graph prompt tuning.

Supported prompts:
    SRP       — Spectral Reverse Prompt (main method)
    SRP-NM    — Ablation: SRP without null-space channel
    SRP-NS    — Ablation: SRP without spectral channel
    SRP-NR    — Ablation: SRP without Reverse mechanism (no mask)
    SRP-Bi    — Ablation: SRP with bidirectional (U-shaped) mask

shots argument:
    shots > 0  : few-shot (shots graphs per class)
    shots <= 0 : full-shot (all graphs per class)
"""

import time
import random
import logging
import numpy as np
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn import metrics

from load_data import GraphDownstream, load_graph_data
from model import GIN
from prompt import SRP
from prompt_ablation import SRP_NM, SRP_NS, SRP_NR, SRP_Bi
from logger import Logger


_PROMPT_CLASSES = {
    'SRP':    SRP,
    'SRP-NM': SRP_NM,
    'SRP-NS': SRP_NS,
    'SRP-NR': SRP_NR,
    'SRP-Bi': SRP_Bi,
}

_ALL_PROMPT_TYPES = list(_PROMPT_CLASSES.keys())


class GraphTask():
    def __init__(self, dataset_name, shots, gnn_type, num_layer, hidden_dim,
                 device, pretrain_task, prompt_type, num_prompts, logger,
                 adapter_dim=32, pca_dim=16):
        self.dataset_name = dataset_name
        self.shots = shots
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.pretrain_task = pretrain_task
        self.prompt_type = prompt_type
        self.num_prompts = num_prompts
        self.logger = logger
        self.adapter_dim = adapter_dim
        self.pca_dim = pca_dim

        if dataset_name in ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']:
            self.graph_list, self.input_dim, self.output_dim = load_graph_data(
                dataset_name, data_folder='./data'
            )
            self.train_data, self.test_data = GraphDownstream(
                self.graph_list, shots, test_fraction=0.4
            )
            self.node_features = torch.cat([g.x for g in self.graph_list], dim=0)
        else:
            raise ValueError(
                'Error: invalid dataset name! '
                'Supported: [ENZYMES, DD, NCI1, NCI109, Mutagenicity]'
            )

        self.initialize_model()
        self.initialize_prompt()

    def initialize_model(self):
        if self.gnn_type == 'GIN':
            self.gnn = GIN(num_layer=self.num_layer,
                           input_dim=self.input_dim,
                           hidden_dim=self.hidden_dim)
        else:
            raise ValueError(f"Error: invalid GNN type! Supported: [GIN]")

        if self.pretrain_task is not None:
            pretrained_gnn_file = (
                f'./pretrained_gnns/{self.dataset_name}_{self.pretrain_task}'
                f'_{self.gnn_type}_5.pth'
            )
            self.gnn.load_state_dict(
                torch.load(pretrained_gnn_file, map_location=self.device)
            )

        print(self.gnn)
        self.gnn.to(self.device)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

    def initialize_prompt(self):
        if self.prompt_type not in _ALL_PROMPT_TYPES:
            raise ValueError(
                f"Error: invalid prompt type '{self.prompt_type}'! "
                f"Supported: {_ALL_PROMPT_TYPES}"
            )

        dim_in_list  = [self.input_dim] + [self.hidden_dim] * (self.num_layer - 1)
        dim_out_list = [self.hidden_dim] * self.num_layer

        PromptCls = _PROMPT_CLASSES[self.prompt_type]
        weight_matrices = [
            self.gnn.convs[i].mlp[0].weight.data for i in range(self.num_layer)
        ]
        self.prompt = PromptCls(
            dim_in_list=dim_in_list,
            dim_out_list=dim_out_list,
            weight_matrices=weight_matrices,
            node_features=self.node_features,
            null_pca_dim=self.pca_dim,
            r_shared=self.adapter_dim,
        ).to(self.device)

        if hasattr(self.prompt, 'decomp_summary'):
            print(f"[{self.prompt_type}] layer decomp plan:\n{self.prompt.decomp_summary()}")
        if hasattr(self.prompt, 'count_parameters'):
            print(f"[{self.prompt_type}] trainable prompt parameters: "
                  f"{self.prompt.count_parameters():,}")

    def train(self, batch_size, lr=0.001, decay=0, epochs=100):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(self.test_data,  batch_size=batch_size, shuffle=False)
        learnable_parameters = (
            list(self.classifier.parameters()) + list(self.prompt.parameters())
        )

        optimizer = torch.optim.Adam(learnable_parameters, lr=lr, weight_decay=decay)

        best_test_accuracy = 0.0
        epoch_times = []

        for epoch in range(1, 1 + epochs):
            epoch_start = time.time()

            # --- Training ---
            total_loss = []
            self.gnn.train()
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                emb = self.gnn(data, self.prompt_type, self.prompt, pooling='mean')
                out = self.classifier(emb)
                loss = F.cross_entropy(out, data.y.squeeze())

                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
            train_loss = np.mean(total_loss)

            # --- Evaluation ---
            self.gnn.eval()
            pred_list, label_list, eval_loss = [], [], []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    emb = self.gnn(data, self.prompt_type, self.prompt, pooling='mean')
                    out = self.classifier(emb)
                    loss = F.cross_entropy(out, data.y.squeeze())
                    pred_list.extend(out.argmax(1).tolist())
                    label_list.extend(data.y.squeeze().tolist())
                    eval_loss.append(loss.item())

            test_accuracy = metrics.accuracy_score(y_true=label_list, y_pred=pred_list)
            test_loss = np.mean(eval_loss)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            log_info = ''.join([
                f'| epoch: {epoch:4d} ',
                f'| train_loss: {train_loss:7.5f}',
                f'| test_loss: {test_loss:7.5f}',
                f'| test_accuracy: {test_accuracy:7.5f} ',
                f'| best_accuracy: {best_test_accuracy:7.5f} ',
                f'| epoch_time: {epoch_time:.3f}s |',
            ])
            self.logger.info(log_info)

        avg_epoch_time = np.mean(epoch_times)
        self.logger.info(
            f'| Training complete | avg_epoch_time: {avg_epoch_time:.3f}s '
            f'| total_time: {sum(epoch_times):.1f}s |'
        )
        return best_test_accuracy


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args, seed):
    shots_tag = 'full' if args.shots <= 0 else str(args.shots)
    run_name = (f'{args.dataset_name}_{shots_tag}_{args.pretrain_task}_'
                f'{args.gnn_type}_{args.prompt_type}_r{args.adapter_dim}_m{args.pca_dim}')
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = str(log_dir / f'{run_name}_seed{seed}.log')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = Logger(filename, formatter)
    set_random_seed(seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    task = GraphTask(
        args.dataset_name, args.shots, args.gnn_type, args.num_layer,
        args.hidden_dim, device, args.pretrain_task, args.prompt_type,
        args.num_prompts, logger,
        adapter_dim=args.adapter_dim,
        pca_dim=args.pca_dim,
    )
    best_accuracy = task.train(args.batch_size, epochs=args.epochs)
    return {'seed': seed, 'best_test_accuracy': best_accuracy, 'log': filename}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream task: graph classification')
    parser.add_argument('--dataset_name', type=str, default='NCI1',
                        help='dataset name [ENZYMES, DD, NCI1, NCI109, Mutagenicity]')
    parser.add_argument('--shots', type=int, default=50,
                        help='graphs per class (default: 50; use 0 or negative for full-shot)')
    parser.add_argument('--gnn_type', type=str, default='GIN', help='gnn type')
    parser.add_argument('--num_layer', type=int, default=5, help='GNN layers (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim (default: 128)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--pretrain_task', type=str, default='GraphCL',
                        help='pretrain task [GraphCL, SimGRACE]')
    parser.add_argument('--prompt_type', type=str, default='SRP', choices=_ALL_PROMPT_TYPES)
    parser.add_argument('--num_prompts', type=int, default=5,
                        help='num_prompts (unused, for compatibility)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='epochs (default: 200)')
    parser.add_argument('--adapter_dim', type=int, default=32,
                        help='r_shared bottleneck dim for SRP (default: 32)')
    parser.add_argument('--pca_dim', type=int, default=16,
                        help='null-space PCA dim for SRP (default: 16)')
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--seed', type=int, help='run one seed (useful for parallel jobs)')
    seed_group.add_argument('--seeds', type=int, nargs='+', help='run the specified seeds')
    parser.add_argument('--log_dir', default='log', help='directory for per-seed logs')
    parser.add_argument('--result_dir', default='results', help='directory for machine-readable results')

    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else (args.seeds if args.seeds else list(range(5)))
    if len(set(seeds)) != len(seeds):
        parser.error('seed values must be unique')
    records = [run(args, seed) for seed in seeds]
    accuracies = [row['best_test_accuracy'] for row in records]
    summary = {
        'config': {key: value for key, value in vars(args).items() if key not in ('seed', 'seeds')},
        'metric': 'maximum test accuracy over all training epochs',
        'seeds': seeds,
        'runs': records,
        'mean': mean(accuracies),
        'std_population': pstdev(accuracies),
    }
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    shots_tag = 'full' if args.shots <= 0 else str(args.shots)
    prefix = (f'{args.dataset_name}_{shots_tag}_{args.pretrain_task}_'
              f'{args.gnn_type}_{args.prompt_type}_r{args.adapter_dim}_m{args.pca_dim}_'
              f'seeds-{"-".join(map(str, seeds))}')
    (result_dir / f'{prefix}.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (result_dir / f'{prefix}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed', 'best_test_accuracy', 'log'])
        writer.writeheader()
        writer.writerows(records)
    print(f"Result: {100 * summary['mean']:.2f} ± {100 * summary['std_population']:.2f}% "
          f"over {len(seeds)} seed(s); details: {result_dir / (prefix + '.json')}")
