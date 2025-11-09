"""
Graph data loading utilities for different model types.
"""

from .data_loader import (
    SPECIAL,
    load_examples,
    build_vocab_from_texts,
    TokenDataset,
    collate,
    resolve_split_globs,
    parse_distance_label_from_text,
    parse_query_nodes_from_text,
)

# Lazy imports for torch_geometric dependencies (only needed for graph models)
def _import_graph_datasets():
    """Lazy import of graph dataset classes that require torch_geometric."""
    try:
        from .graph_token_dataset import GraphTokenDataset, add_query_encoding_to_features
        from .graph_token_dataset_autograph import GraphTokenDatasetForAutoGraph
        return GraphTokenDataset, add_query_encoding_to_features, GraphTokenDatasetForAutoGraph
    except ImportError as e:
        if "torch_geometric" in str(e):
            raise ImportError(
                "GraphTokenDataset requires torch_geometric. "
                "Install it with: pip install torch-geometric"
            ) from e
        raise

# Make them available via lazy import
def __getattr__(name):
    if name in ['GraphTokenDataset', 'add_query_encoding_to_features', 'GraphTokenDatasetForAutoGraph']:
        GraphTokenDataset, add_query_encoding_to_features, GraphTokenDatasetForAutoGraph = _import_graph_datasets()
        globals()['GraphTokenDataset'] = GraphTokenDataset
        globals()['add_query_encoding_to_features'] = add_query_encoding_to_features
        globals()['GraphTokenDatasetForAutoGraph'] = GraphTokenDatasetForAutoGraph
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'SPECIAL',
    'load_examples',
    'build_vocab_from_texts',
    'TokenDataset',
    'collate',
    'resolve_split_globs',
    'parse_distance_label_from_text',
    'parse_query_nodes_from_text',
    'GraphTokenDataset',
    'add_query_encoding_to_features',
    'GraphTokenDatasetForAutoGraph',
]
