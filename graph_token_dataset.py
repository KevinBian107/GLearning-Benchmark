"""PyG Dataset adapter for graph-token JSON files"""

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

    # Parse edges: look for <e> followed by two integers
    i = 0
    while i < len(tokens):
        if tokens[i] == '<e>' and i + 2 < len(tokens):
            try:
                src = int(tokens[i + 1])
                tgt = int(tokens[i + 2])
                edges.append((src, tgt))
                i += 3
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


def parse_label_from_text(text: str) -> Optional[int]:
    """Parse label from tokenized graph text.

    Args:
        text: Tokenized graph string ending with "<p> yes/no <eos>"

    Returns:
        Binary label (1 for yes, 0 for no) or None if not found
    """
    tokens = text.split()
    # Look for <p> tag followed by yes/no
    for i, tok in enumerate(tokens):
        if tok == '<p>' and i + 1 < len(tokens):
            label_tok = tokens[i + 1].lower()
            if label_tok == 'yes':
                return 1
            elif label_tok == 'no':
                return 0
    return None


class GraphTokenDataset(InMemoryDataset):
    """PyG Dataset for graph-token generated graphs.

    Loads graphs from JSON files containing tokenized graph representations
    and converts them to PyG Data objects.
    """

    def __init__(
        self,
        root: str,
        task: str = 'cycle_check',
        algorithm: str = 'er',
        split: str = 'train',
        use_split_tasks_dirs: bool = True,
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
        """
        self.task = task
        self.algorithm = algorithm
        self.split = split
        self.use_split_tasks_dirs = use_split_tasks_dirs
        self._root = root

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        """Directory containing raw JSON files."""
        if self.use_split_tasks_dirs:
            if self.split == 'test':
                base = os.path.join(self._root, 'tasks_test', self.task, self.algorithm)
            else:
                base = os.path.join(self._root, 'tasks_train', self.task, self.algorithm)
        else:
            base = os.path.join(self._root, 'tasks', self.task, self.algorithm)
        return os.path.join(base, self.split)

    @property
    def processed_dir(self) -> str:
        """Directory to save processed data."""
        base_name = f"{self.task}_{self.algorithm}_{self.split}"
        if self.use_split_tasks_dirs:
            base_name += "_split"
        return os.path.join(self._root, 'processed', base_name)

    @property
    def raw_file_names(self) -> List[str]:
        """List of raw file names (not used since we process in-memory)."""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """List of processed file names."""
        return ['data.pt']

    def download(self):
        """Download not needed - assumes graph-token already generated data."""
        pass

    def process(self):
        """Process JSON files into PyG Data objects."""
        # Find all JSON files in raw directory
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
                # Extract text and label
                text = record.get('text', '')
                if not text:
                    continue

                # Parse graph structure from text
                nodes, edges = parse_graph_from_text(text)

                if len(nodes) == 0:
                    # If no explicit node list, infer from edges
                    node_set = set()
                    for src, tgt in edges:
                        node_set.add(src)
                        node_set.add(tgt)
                    nodes = sorted(list(node_set))

                if len(nodes) == 0:
                    continue  # Skip empty graphs

                # Parse label
                label = record.get('label')
                if label is None:
                    label = parse_label_from_text(text)
                if label is None:
                    continue  # Skip if no label found

                # Convert to PyG format
                num_nodes = max(nodes) + 1 if nodes else 0

                # Create edge_index tensor
                if len(edges) > 0:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                else:
                    # Empty graph - no edges
                    edge_index = torch.empty((2, 0), dtype=torch.long)

                # Create node features (one-hot encoding of node IDs)
                # For synthetic graphs, we use constant features or one-hot
                x = torch.ones((num_nodes, 1), dtype=torch.float)  # Constant features

                # Create edge attributes (dummy - constant vectors for unweighted graphs)
                # GINE requires edge features to match node feature dim after encoding
                # We use 64D to match the gnn.dim_inner from config
                num_edges = edge_index.size(1)
                edge_attr = torch.ones((num_edges, 64), dtype=torch.float)

                # Create PyG Data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,  # Add edge attributes
                    y=torch.tensor([label], dtype=torch.long),
                    num_nodes=num_nodes,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
