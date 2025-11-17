"""
PyG Dataset adapter for graph-token JSON files.
This creates PyG Data objects that can be used with AutoGraph's tokenizer.
"""

import os
import json
from glob import glob
from typing import List, Optional, Tuple
import torch
from torch_geometric.data import Data, InMemoryDataset


def parse_graph_from_text(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Parse nodes and edges from tokenized graph text.

    Args:
        text: Tokenized graph string like "<bos> 0 1 <e> 0 7 <e> 1 6 ... <n> 0 1 2 ..."

    Returns:
        nodes: List of node IDs
        edges: List of (source, target) tuples
    """
    tokens = text.split()
    edges = []
    nodes = []

    # Parse edges: look for pattern "src dst <e>"
    i = 0
    while i < len(tokens):
        # Check for pattern: number number <e>
        if i + 2 < len(tokens) and tokens[i+2] == '<e>':
            try:
                src = int(tokens[i])
                tgt = int(tokens[i + 1])
                edges.append((src, tgt))
                i += 3  # Skip to next edge
            except ValueError:
                i += 1
        elif tokens[i] == '<n>' and i + 1 < len(tokens):
            # Parse nodes after <n> tag
            i += 1
            while i < len(tokens) and tokens[i] not in ['<q>', '<p>', '<eos>']:
                try:
                    node_id = int(tokens[i])
                    nodes.append(node_id)
                    i += 1
                except ValueError:
                    break
            break
        else:
            i += 1

    return nodes, edges


def parse_query_nodes_from_text(text: str) -> Optional[Tuple[int, int]]:
    """Parse query nodes from text like '<q> shortest_distance 2 5' -> (2, 5)

    Args:
        text: Tokenized graph string with query

    Returns:
        Tuple of (u, v) node IDs, or None if not found
    """
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok == '<q>' and i + 3 < len(tokens):
            # Format: <q> shortest_distance u v
            if tokens[i + 1] == 'shortest_distance':
                try:
                    u = int(tokens[i + 2])
                    v = int(tokens[i + 3])
                    return (u, v)
                except ValueError:
                    pass
    return None


def parse_label_from_text(text: str, task: str = 'cycle_check') -> Optional[int]:
    """Parse label from tokenized graph text.

    Args:
        text: Tokenized graph string ending with "<p> yes/no <eos>" or "<p> lenX <eos>"
        task: Task name (cycle_check, shortest_path, etc.)

    Returns:
        Label as integer, or None if not found
    """
    tokens = text.split()
    # Look for <p> tag followed by label
    for i, tok in enumerate(tokens):
        if tok == '<p>' and i + 1 < len(tokens):
            label_tok = tokens[i + 1].upper()  # Handle case-insensitive

            # Handle cycle_check (yes/no)
            if label_tok in ('YES', 'NO'):
                return 1 if label_tok == 'YES' else 0

            # Handle shortest_path (len1-len7)
            if label_tok.startswith('LEN'):
                try:
                    distance = int(label_tok[3:])  # 'len3' -> 3
                    # Convert to 0-indexed for PyTorch (len1->0, len2->1, ..., len7->6)
                    return distance - 1
                except ValueError:
                    pass

            # Handle unreachable (INF) for shortest_path
            if label_tok in ('INF', 'INFINITY'):
                return None  # Skip unreachable samples

    return None


def parse_graph_from_json(record: dict, task: str = 'cycle_check') -> Tuple[List[Tuple[int, int]], int, Optional[int]]:
    """Parse graph structure from graph-token JSON record.

    Args:
        record: Dictionary with 'text' and optionally 'nodes', 'edges', 'label' keys
        task: Task name (cycle_check, shortest_path, etc.)

    Returns:
        edges: List of (source, target) tuples
        num_nodes: Number of nodes in the graph
        label: Label as integer (task-dependent)
    """
    # Try to get nodes and edges from direct fields first
    nodes = record.get('nodes', [])
    edges = record.get('edges', [])

    # If not available, parse from text
    if not edges:
        text = record.get('text', '')
        if text:
            nodes, edges = parse_graph_from_text(text)

    # Get label
    label = record.get('label')
    if label is None:
        # Try parsing from text
        text = record.get('text', '')
        if text:
            label = parse_label_from_text(text, task=task)

    # Determine number of nodes
    if nodes:
        num_nodes = max(nodes) + 1 if nodes else 0
    elif edges:
        node_set = set()
        for src, tgt in edges:
            node_set.add(src)
            node_set.add(tgt)
        num_nodes = max(node_set) + 1 if node_set else 0
    else:
        num_nodes = 0

    return edges, num_nodes, label


class GraphTokenDatasetForAutoGraph(InMemoryDataset):
    """PyG Dataset for graph-token generated graphs (for use with AutoGraph tokenizer).

    Loads graphs from JSON files and converts them to PyG Data objects
    that can be tokenized by AutoGraph's Graph2TrailTokenizer.
    """

    def __init__(
        self,
        root: str,
        task: str = 'cycle_check',
        algorithm: str = 'er',
        split: str = 'train',
        use_split_tasks_dirs: bool = True,
        data_fraction: float = 1.0,
        seed: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Args:
            root: Root directory (graph-token repo path)
            task: Task name (cycle_check, connected_nodes, etc.)
            algorithm: Graph generation algorithm (er, ba, sbm, etc.)
            split: Data split (train, val, test)
            use_split_tasks_dirs: Use tasks_train/tasks_test structure
            data_fraction: Fraction of data to use (0.0-1.0)
            seed: Random seed for reproducible sampling
        """
        self.task = task
        self.algorithm = algorithm
        self.split = split
        self.use_split_tasks_dirs = use_split_tasks_dirs
        self.data_fraction = max(0.0, min(1.0, data_fraction))  # Clamp to [0, 1]
        self.seed = seed
        self._root = root

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        """Directory containing raw JSON files."""
        if self.use_split_tasks_dirs:
            if self.split in ['val', 'test']:
                base = os.path.join(self._root, 'tasks_test', self.task, self.algorithm)
            else:
                base = os.path.join(self._root, 'tasks_train', self.task, self.algorithm)
        else:
            base = os.path.join(self._root, 'tasks', self.task, self.algorithm)

        split_dir = os.path.join(base, self.split)

        # For validation, check if it exists, otherwise use test
        if self.split == 'val' and self.use_split_tasks_dirs:
            json_pattern = os.path.join(split_dir, '*.json')
            if len(glob(json_pattern)) == 0:
                split_dir = os.path.join(base, 'test')

        return split_dir

    @property
    def processed_dir(self) -> str:
        """Directory to save processed data."""
        base_name = f"autograph_{self.task}_{self.algorithm}_{self.split}"
        if self.use_split_tasks_dirs:
            base_name += "_split"
        if self.data_fraction < 1.0:
            # Include fraction in cache key to avoid conflicts
            base_name += f"_frac{self.data_fraction:.2f}"
        return os.path.join(self._root, 'processed', base_name)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self):
        """Download not needed - assumes graph-token already generated data."""
        pass

    def process(self):
        """Process JSON files into PyG Data objects."""
        json_pattern = os.path.join(self.raw_dir, '*.json')
        json_files = sorted(glob(json_pattern))

        if len(json_files) == 0:
            raise RuntimeError(
                f"No JSON files found at {json_pattern}. "
                f"Did you run the graph-token task generator?"
            )

        data_list = []

        for json_file in json_files:
            with open(json_file, 'r') as f:
                content = json.load(f)

            # Handle both list and single dict formats
            if isinstance(content, list):
                records = content
            else:
                records = [content]

            for record in records:
                # Parse graph structure from JSON
                edges, num_nodes, label = parse_graph_from_json(record, task=self.task)

                if num_nodes == 0:
                    continue  # Skip empty graphs

                if label is None:
                    continue  # Skip if no label found

                # Parse query nodes for shortest_path task
                query_nodes = None
                if self.task == 'shortest_path':
                    text = record.get('text', '')
                    if text:
                        query_nodes = parse_query_nodes_from_text(text)

                # Create edge_index tensor
                if len(edges) > 0:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                else:
                    # Empty graph - no edges
                    edge_index = torch.empty((2, 0), dtype=torch.long)

                # Create PyG Data object (minimal - AutoGraph doesn't need node features)
                data = Data(
                    edge_index=edge_index,
                    y=torch.tensor([label], dtype=torch.long),
                    num_nodes=num_nodes,
                )

                # Add query nodes if available
                if query_nodes is not None:
                    data.query_u = query_nodes[0]
                    data.query_v = query_nodes[1]

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        # Apply data fraction sampling if needed
        if self.data_fraction < 1.0 and len(data_list) > 0:
            import random
            rng = random.Random(self.seed)
            original_size = len(data_list)
            n_samples = max(1, int(original_size * self.data_fraction))
            data_list = rng.sample(data_list, n_samples)
            print(f"[{self.split}] Sampled {n_samples}/{original_size} graphs (fraction={self.data_fraction})")

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
