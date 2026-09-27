import random
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Flickr, TUDataset


def load_graph_data(dataset_name, data_folder):
    if dataset_name in ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']:
        dataset = TUDataset(f'{data_folder}/TUDataset/', name=dataset_name)
    else:
        raise ValueError('Error: invalid dataset name! Supported datasets: [ENZYMES, DD, NCI1, NCI109, Mutagenicity]')

    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    return dataset, input_dim, output_dim


def GraphDownstream(data, shots=5, test_fraction=0.4):
    """Build train/test graph splits.

    Args:
        shots: number of graphs per class.
               Use shots <= 0 for full-shot (all graphs per class as training).
    """
    num_classes = data.y.max().item() + 1
    graph_list = []
    for c in range(num_classes):
        indices = torch.where(data.y.squeeze() == c)[0].tolist()
        if shots <= 0:
            graph_list.extend(indices)          # full-shot: use all graphs
        elif len(indices) < shots:
            graph_list.extend(indices)
        else:
            graph_list.extend(random.sample(indices, k=shots))
    random_graph_list = random.sample(range(len(data)), k=len(data))
    for graph in graph_list:
        random_graph_list.remove(graph)
    train_data = [data[g] for g in graph_list]
    test_data = [data[g] for g in random_graph_list[:int(test_fraction * len(data))]]

    return train_data, test_data
