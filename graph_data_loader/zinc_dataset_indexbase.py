"""
ZINC dataset tokenization for IBTT (Index-Based Tokenization Transformer).

IMPORTANT: This uses MOLECULAR tokenization (atom types, bond types),
NOT structure-only tokenization like synthetic graphs.

Loads ZINC molecular graphs and tokenizes them preserving chemical information.

NEW: Uses FIXED vocabulary from zinc_vocab.py to ensure consistent feature
encoding with AGTT (same embedding indices for same chemical features).
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import ZINC
from .zinc_vocab import (
    ZINC_ATOM_TYPES as ATOM_TYPE_LIST,
    ZINC_BOND_TYPES as BOND_TYPE_LIST,
    get_atom_type_id,
    get_bond_type_id,
)


# ZINC atom type mapping (from integer index to atom symbol)
# Using the fixed list from zinc_vocab.py
ZINC_ATOM_TYPES = {i: atom for i, atom in enumerate(ATOM_TYPE_LIST)}

# ZINC bond type mapping (from integer index to bond type)
# Note: Bond type indices in PyG ZINC are 1-based (1=single, 2=double, 3=triple, 4=aromatic)
ZINC_BOND_TYPES = {
    0: 'single',    # Not used (bonds start at index 1)
    1: 'single',    # Single bond
    2: 'double',    # Double bond
    3: 'triple',    # Triple bond
    4: 'aromatic',  # Aromatic bond
}


class ZINCTokenizationDataset(Dataset):
    """
    ZINC dataset with on-the-fly molecular tokenization for transformer models.

    KEY DIFFERENCE from synthetic graphs:
    - Synthetic graphs: Structure-only (node indices: 0, 1, 2, ...)
    - ZINC molecules: Chemistry-aware (atom types: C, N, O, bond types: single, double, ...)

    Tokenization format:
    <bos> <atom> C <atom> N <bond> single 0 1 <atom> O <bond> double 1 2 ... <q> regression <p> val_X_XX <eos>

    Where:
    - <atom> C = atom at index 0 is Carbon
    - <bond> single 0 1 = single bond between atoms 0 and 1
    - Preserves full chemical structure information

    Args:
        zinc_root: Root directory for ZINC dataset
        split: 'train', 'val', or 'test'
        subset: If True, use 12K subset; else use full dataset
        max_vocab: Maximum vocabulary size (for compatibility, not enforced here)
        max_len: Maximum sequence length (for truncation)
    """

    def __init__(
        self,
        zinc_root: str = './data/ZINC',
        split: str = 'train',
        subset: bool = True,
        max_vocab: int = 10000,
        max_len: int = 2048,
    ):
        super().__init__()

        self.zinc_root = zinc_root
        self.split = split
        self.subset = subset
        self.max_len = max_len

        # Load ZINC dataset
        self.zinc_dataset = ZINC(root=zinc_root, subset=subset, split=split)

        print(f"Loaded ZINC {split} split: {len(self.zinc_dataset)} molecules")

    def __len__(self):
        return len(self.zinc_dataset)

    def decode_atom_features(self, node_features):
        """
        Decode ZINC node features to atom types.

        ZINC node features are integers (shape [num_nodes, 1]):
        - Each value is an atom type index (0=C, 1=N, 2=O, etc.)

        Args:
            node_features: Tensor of shape [num_nodes, 1]

        Returns:
            List of atom type strings (e.g., ['C', 'N', 'O', ...])
        """
        atom_types = []

        # Node features are single integers (atom type indices)
        for node_feat in node_features:
            idx = node_feat.item() if node_feat.numel() == 1 else node_feat[0].item()
            atom_type = ZINC_ATOM_TYPES.get(idx, 'X')  # 'X' for unknown
            atom_types.append(atom_type)

        return atom_types

    def decode_bond_features(self, edge_attr):
        """
        Decode ZINC edge features to bond types.

        ZINC edge features are integers (shape [num_edges]):
        - 1 = single bond
        - 2 = double bond
        - 3 = triple bond
        - 4 = aromatic bond

        Args:
            edge_attr: Tensor of shape [num_edges]

        Returns:
            List of bond type strings
        """
        bond_types = []

        # Edge attributes are single integers (bond type indices)
        for edge_feat in edge_attr:
            bond_idx = edge_feat.item()
            if bond_idx == 1:
                bond_types.append('single')
            elif bond_idx == 2:
                bond_types.append('double')
            elif bond_idx == 3:
                bond_types.append('triple')
            elif bond_idx == 4:
                bond_types.append('aromatic')
            else:
                bond_types.append('unknown')

        return bond_types

    def tokenize_molecule(self, data, label):
        """
        Tokenize a ZINC molecule preserving chemical information.

        Format:
        <bos>
        <atom> C <atom> N <atom> O ...           [atom types]
        <bond> single 0 1 <bond> double 1 2 ...  [bond types with indices]
        <q> regression
        <p> val_4_23
        <eos>

        Args:
            data: PyG Data object with x (node features), edge_index, edge_attr
            label: Target value (float)

        Returns:
            Tokenized string with chemical information
        """
        tokens = ["<bos>"]

        # Decode atom types from node features
        atom_types = self.decode_atom_features(data.x)

        # Add atoms with their types
        for i, atom_type in enumerate(atom_types):
            tokens.extend(["<atom>", atom_type])

        # Decode bond types from edge attributes
        bond_types = self.decode_bond_features(data.edge_attr)

        # Add bonds with their types and indices
        # Note: ZINC edge_index is undirected, so we need to handle duplicates
        seen_edges = set()
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            # Canonical edge representation (smaller index first)
            edge_tuple = tuple(sorted([u, v]))

            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                bond_type = bond_types[i] if i < len(bond_types) else 'unknown'
                tokens.extend(["<bond>", bond_type, str(u), str(v)])

        # Add task query
        tokens.extend(["<q>", "regression"])

        # Add prediction label
        # For ZINC, y is a continuous value (constrained solubility)
        # Format: val_4_23 for 4.23, val_neg2_10 for -2.10
        label_str = f"val_{label:.2f}".replace('.', '_').replace('-', 'neg')
        tokens.extend(["<p>", label_str, "<eos>"])

        return " ".join(tokens)

    def __getitem__(self, idx):
        """
        Get a tokenized molecular example.

        Returns:
            dict with:
                - text: tokenized sequence with chemistry (str)
                - label: target value (float)
                - graph_id: index (for debugging)
        """
        # Get PyG data object
        data = self.zinc_dataset[idx]

        # Extract label
        label = data.y.item()  # Convert tensor to float

        # Tokenize molecule with chemical information
        text = self.tokenize_molecule(data, label)

        # Truncate if needed (preserve <bos> and <eos>)
        tokens = text.split()
        if len(tokens) > self.max_len:
            # Keep <bos> at start and <eos> at end
            tokens = tokens[:self.max_len-1] + ['<eos>']
            text = " ".join(tokens)

        return {
            'text': text,
            'label': label,
            'graph_id': f"zinc_{self.split}_{idx}",
        }


def collate_zinc_batch(batch, pad_id):
    """
    Collate function for ZINC tokenization dataset.
    Similar to standard collate but handles regression labels.

    NOTE: This is not actually used - we use the standard TokenDataset + collate
    from data_loader.py which handles the vocabulary properly.

    Args:
        batch: List of dicts with 'text' and 'label' keys
        pad_id: Padding token ID

    Returns:
        tuple of (token_ids, attention_mask, labels)
    """
    from graph_data_loader import build_vocab_from_texts

    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return texts, labels_tensor
