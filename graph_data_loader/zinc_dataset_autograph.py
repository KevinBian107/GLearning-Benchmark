"""
ZINC dataset for AGTT (AutoGraph Trail Tokenization + Transformer).

Loads ZINC molecular graphs and prepares them for AutoGraph's trail tokenization.
The tokenizer will encode atom types and bond types into trail sequences.
"""

import torch
from torch_geometric.datasets import ZINC
from torch.utils.data import Dataset


class ZINCDatasetForAutoGraph(Dataset):
    """
    ZINC dataset wrapper for AutoGraph tokenization.

    Unlike IBTT which uses pre-tokenized sequences, AGTT loads native PyG graphs
    and applies trail-based tokenization on-the-fly using AutoGraph's Graph2TrailTokenizer.

    The tokenizer (when labeled_graph=True) will:
    - Encode atom types from data.x into the trail
    - Encode bond types from data.edge_attr into the trail
    - Create random walk sequences that preserve chemical information

    Args:
        zinc_root: Root directory for ZINC dataset
        split: 'train', 'val', or 'test'
        subset: If True, use 12K subset; else use full dataset
    """

    def __init__(
        self,
        zinc_root: str = './data/ZINC',
        split: str = 'train',
        subset: bool = True,
    ):
        super().__init__()

        self.zinc_root = zinc_root
        self.split = split
        self.subset = subset

        # Load ZINC dataset from PyTorch Geometric
        self.zinc_dataset = ZINC(root=zinc_root, subset=subset, split=split)

        print(f"Loaded ZINC {split} split: {len(self.zinc_dataset)} molecules")

    def __len__(self):
        return len(self.zinc_dataset)

    def __getitem__(self, idx):
        """
        Return PyG Data object directly for AutoGraph tokenization.

        The data object contains:
        - x: Node features [num_nodes, 1] - atom type indices (0-8)
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Edge features [num_edges] - bond type indices (1-4)
        - y: Target value (scalar regression)
        """
        data = self.zinc_dataset[idx]

        # Ensure data is in the correct format for AutoGraph
        # x should be flattened to [num_nodes] for labeled graph tokenization
        if data.x.dim() == 2 and data.x.size(1) == 1:
            # Already in correct format [num_nodes, 1], will be flattened by tokenizer
            pass

        # edge_attr should be 1D [num_edges]
        if data.edge_attr.dim() == 2 and data.edge_attr.size(1) == 1:
            data.edge_attr = data.edge_attr.flatten()

        return data


def get_zinc_num_types():
    """
    Return the number of node types (atoms) and edge types (bonds) in ZINC.

    ZINC atom types (9 types):
    - 0: C (Carbon)
    - 1: N (Nitrogen)
    - 2: O (Oxygen)
    - 3: F (Fluorine)
    - 4: P (Phosphorus)
    - 5: S (Sulfur)
    - 6: Cl (Chlorine)
    - 7: Br (Bromine)
    - 8: I (Iodine)

    ZINC bond types (4 types):
    - 1: Single bond
    - 2: Double bond
    - 3: Triple bond
    - 4: Aromatic bond

    Returns:
        (num_node_types, num_edge_types)
    """
    return 9, 4  # 9 atom types, 4 bond types
