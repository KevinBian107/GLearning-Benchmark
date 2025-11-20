"""PyG Dataset adapter for graph-token JSON files"""

import os
import json
from glob import glob
from typing import List, Optional, Tuple
import torch
from torch_geometric.data import Data, InMemoryDataset

# Import parsing functions from data_loader
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_data_loader.data_loader import parse_distance_label_from_text, parse_query_nodes_from_text


def add_query_encoding_to_features(x: torch.Tensor, query_u: int, query_v: int) -> torch.Tensor:
    """Add binary query encodings to node features for shortest_path task.

    Args:
        x: Node features [num_nodes, feature_dim]
        query_u: Source node ID
        query_v: Target node ID

    Returns:
        Augmented features [num_nodes, feature_dim + 2] with binary flags for source/target
    """
    num_nodes = x.size(0)
    query_encoding = torch.zeros(num_nodes, 2, dtype=x.dtype, device=x.device)
    query_encoding[query_u, 0] = 1.0  # Mark source node
    query_encoding[query_v, 1] = 1.0  # Mark target node
    return torch.cat([x, query_encoding], dim=1)


class AddQueryEncoding:
    """Transform that adds query node encoding to node features for shortest_path task."""

    def __call__(self, data):
        """
        Add query encoding to data.x if query nodes are present.

        Args:
            data: PyG Data object

        Returns:
            Modified Data object with query encoding added to x
        """
        if hasattr(data, 'query_u') and hasattr(data, 'query_v'):
            query_u = data.query_u.item() if data.query_u.dim() == 0 else data.query_u[0].item()
            query_v = data.query_v.item() if data.query_v.dim() == 0 else data.query_v[0].item()
            data.x = add_query_encoding_to_features(data.x, query_u, query_v)
        return data


def parse_graph_from_text(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Parse nodes and edges from tokenized graph text.

    Args:
        text: Tokenized graph string like "<bos> 0 1 <e> 0 7 <e> 1 6 ... <n> 0 1 2 ..."
              Format: edge pairs come BEFORE <e> marker: "src dst <e> src dst <e> ..."

    Returns:
        nodes: List of node IDs
        edges: List of (source, target) tuples
    """
    tokens = text.split()
    edges = []
    nodes = []

    # Parse edges: look for pattern "src dst <e>"
    # The two numbers come BEFORE <e>, not after!
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
        algorithm: Optional[List[str]] = None,
        split: str = 'train',
        use_split_tasks_dirs: bool = True,
        seed: int = 0,
        num_graphs: Optional[int] = None,
        num_pairs_per_graph: Optional[int] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Args:
            root: Root directory (graph-token repo path)
            task: Task name (cycle_check, shortest_path, etc.)
            algorithm: Graph generation algorithm(s) - single string or list of strings
            split: Data split (train, val, test)
            use_split_tasks_dirs: Use tasks_train/tasks_test structure
            seed: Random seed for reproducible sampling
            num_graphs: Number of graph files to sample (per algorithm if multiple)
            num_pairs_per_graph: For shortest_path, number of query pairs to sample per graph
        """
        self.task = task
        # Support both single algorithm (backward compat) and list of algorithms
        if isinstance(algorithm, str):
            self.algorithms = [algorithm]
        elif algorithm is None:
            self.algorithms = ['er']  # Default
        else:
            self.algorithms = algorithm
        self.algorithm = self.algorithms[0]  # For backward compatibility
        self.split = split
        self.use_split_tasks_dirs = use_split_tasks_dirs
        self.seed = seed
        self.num_graphs = num_graphs
        self.num_pairs_per_graph = num_pairs_per_graph
        self._root = root

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        """Directory containing raw JSON files."""
        if self.use_split_tasks_dirs:
            if self.split in ['val', 'test']:
                # Both val and test should use tasks_test directory
                base = os.path.join(self._root, 'tasks_test', self.task, self.algorithm)
            else:
                base = os.path.join(self._root, 'tasks_train', self.task, self.algorithm)
        else:
            base = os.path.join(self._root, 'tasks', self.task, self.algorithm)

        split_dir = os.path.join(base, self.split)

        # For validation, look in test directory since val is stored there
        if self.split == 'val' and self.use_split_tasks_dirs:
            # Check if val exists in tasks_test, if not use test as val
            json_pattern = os.path.join(split_dir, '*.json')
            if len(glob(json_pattern)) == 0:
                print(f"[warn] No validation directory found, using test split for validation")
                split_dir = os.path.join(base, 'test')

        return split_dir

    @property
    def processed_dir(self) -> str:
        """Directory to save processed data."""
        # Create unique cache key based on algorithms and sampling params
        algo_str = '+'.join(sorted(self.algorithms))
        base_name = f"{self.task}_{algo_str}_{self.split}"
        if self.use_split_tasks_dirs:
            base_name += "_split"
        if self.num_graphs is not None:
            base_name += f"_ng{self.num_graphs}"
        if self.num_pairs_per_graph is not None:
            base_name += f"_np{self.num_pairs_per_graph}"
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
        import random

        # Collect JSON files from all algorithms
        all_json_files = []
        for algo in self.algorithms:
            # Build path for this algorithm
            if self.use_split_tasks_dirs:
                if self.split in ['val', 'test']:
                    base = os.path.join(self._root, 'tasks_test', self.task, algo)
                else:
                    base = os.path.join(self._root, 'tasks_train', self.task, algo)
            else:
                base = os.path.join(self._root, 'tasks', self.task, algo)

            split_dir = os.path.join(base, self.split)

            # For validation, check if val exists, otherwise use test
            if self.split == 'val' and self.use_split_tasks_dirs:
                json_pattern_check = os.path.join(split_dir, '*.json')
                if len(glob(json_pattern_check)) == 0:
                    split_dir = os.path.join(base, 'test')

            json_pattern = os.path.join(split_dir, '*.json')
            algo_files = sorted(glob(json_pattern))

            # Sample num_graphs from this algorithm if specified
            if self.num_graphs is not None and len(algo_files) > self.num_graphs:
                rng = random.Random(self.seed + hash(algo) % 10000)
                algo_files = rng.sample(algo_files, self.num_graphs)
                algo_files = sorted(algo_files)
                print(f"  [{algo}] Sampled {self.num_graphs}/{len(sorted(glob(json_pattern)))} graph files")

            all_json_files.extend(algo_files)

        if len(all_json_files) == 0:
            raise RuntimeError(
                f"No JSON files found for algorithms {self.algorithms}. "
                f"Did you run the graph-token task generator?"
            )

        print(f"[GraphTokenDataset] Processing {len(all_json_files)} graph files from {len(self.algorithms)} algorithm(s)")

        data_list = []
        rng = random.Random(self.seed)

        for json_file in all_json_files:
            with open(json_file, 'r') as f:
                content = json.load(f)

            # Handle both list and single dict formats
            if isinstance(content, list):
                records = content
            else:
                records = [content]

            # For shortest_path with num_pairs_per_graph, collect all pairs from this graph first
            if self.task == 'shortest_path' and self.num_pairs_per_graph is not None:
                graph_pairs = []
                for record in records:
                    text = record.get('text', '')
                    if not text:
                        continue

                    # Parse graph structure
                    nodes, edges = parse_graph_from_text(text)
                    if len(nodes) == 0:
                        node_set = set()
                        for src, tgt in edges:
                            node_set.add(src)
                            node_set.add(tgt)
                        nodes = sorted(list(node_set))
                    if len(nodes) == 0:
                        continue

                    # Parse label and query nodes
                    label = record.get('label')
                    if label is None:
                        label = parse_distance_label_from_text(text)
                    query_nodes = parse_query_nodes_from_text(text)
                    if query_nodes is None or label is None:
                        continue

                    graph_pairs.append((nodes, edges, label, query_nodes))

                # Sample pairs from this graph
                if len(graph_pairs) > self.num_pairs_per_graph:
                    graph_pairs = rng.sample(graph_pairs, self.num_pairs_per_graph)

                # Process sampled pairs
                for nodes, edges, label, query_nodes in graph_pairs:
                    query_u, query_v = query_nodes
                    num_nodes = max(nodes) + 1 if nodes else 0

                    if len(edges) > 0:
                        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)

                    x = torch.ones((num_nodes, 1), dtype=torch.float)

                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        y=torch.tensor([label], dtype=torch.long),
                        num_nodes=num_nodes,
                        query_u=torch.tensor([query_u], dtype=torch.long),
                        query_v=torch.tensor([query_v], dtype=torch.long),
                    )

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                continue  # Skip regular processing for this file

            # Regular processing for cycle_check or when num_pairs_per_graph is not set
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

                # Parse label based on task
                label = record.get('label')
                query_u, query_v = None, None

                if self.task == 'shortest_path':
                    # Parse distance label and query nodes
                    if label is None:
                        label = parse_distance_label_from_text(text)
                    query_nodes = parse_query_nodes_from_text(text)
                    if query_nodes is not None:
                        query_u, query_v = query_nodes
                    if label is None or query_nodes is None:
                        continue  # Skip if missing query or label
                else:
                    # Binary tasks like cycle_check
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

                # Create node features (constant features for now)
                # For shortest_path, query encodings will be added by pre_transform
                x = torch.ones((num_nodes, 1), dtype=torch.float)

                # Create PyG Data object
                # Don't create edge_attr for unweighted graphs - let model handle if needed
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor([label], dtype=torch.long),
                    num_nodes=num_nodes,
                )

                # Add query nodes for shortest_path task
                if query_u is not None and query_v is not None:
                    data.query_u = torch.tensor([query_u], dtype=torch.long)
                    data.query_v = torch.tensor([query_v], dtype=torch.long)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        print(f"[GraphTokenDataset] Processed {len(data_list)} data samples")

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
