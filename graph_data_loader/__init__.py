"""
Graph data loading utilities for different model types.
"""

from .data_loader import (
    SPECIAL,
    load_examples,
    load_examples_multi_algorithm,
    determine_num_classes,
    determine_num_classes_pyg,
    build_vocab_from_texts,
    TokenDataset,
    collate,
    resolve_split_globs,
    resolve_multi_algorithm_globs,
    parse_distance_label_from_text,
    parse_query_nodes_from_text,
    balance_classes,
    get_balanced_indices,
)

# ZINC fixed vocabulary for consistent feature encoding
from .zinc_vocab import (
    build_fixed_zinc_vocab,
    get_atom_type_id,
    get_bond_type_id,
    get_atom_type_from_id,
    get_bond_type_from_id,
    extend_vocab_with_dynamic_tokens,
    map_autograph_token_to_fixed_id,
    ZINC_ATOM_TYPES,
    ZINC_BOND_TYPES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
)

# Lazy imports for torch_geometric dependencies (only needed for graph models)
def _import_graph_datasets():
    """Lazy import of graph dataset classes that require torch_geometric."""
    try:
        from .graph_token_dataset_nativegraph import GraphTokenDataset, add_query_encoding_to_features, AddQueryEncoding
        from .graph_token_dataset_autograph import GraphTokenDatasetForAutoGraph
        from .zinc_dataset_indexbase import ZINCTokenizationDataset, collate_zinc_batch
        from .zinc_dataset_autograph import ZINCDatasetForAutoGraph, get_zinc_num_types
        return GraphTokenDataset, add_query_encoding_to_features, AddQueryEncoding, GraphTokenDatasetForAutoGraph, ZINCTokenizationDataset, collate_zinc_batch, ZINCDatasetForAutoGraph, get_zinc_num_types
    except ImportError as e:
        if "torch_geometric" in str(e):
            raise ImportError(
                "GraphTokenDataset requires torch_geometric. "
                "Install it with: pip install torch-geometric"
            ) from e
        raise

# Make them available via lazy import
def __getattr__(name):
    if name in ['GraphTokenDataset', 'add_query_encoding_to_features', 'AddQueryEncoding', 'GraphTokenDatasetForAutoGraph', 'ZINCTokenizationDataset', 'collate_zinc_batch', 'ZINCDatasetForAutoGraph', 'get_zinc_num_types']:
        GraphTokenDataset, add_query_encoding_to_features, AddQueryEncoding, GraphTokenDatasetForAutoGraph, ZINCTokenizationDataset, collate_zinc_batch, ZINCDatasetForAutoGraph, get_zinc_num_types = _import_graph_datasets()
        globals()['GraphTokenDataset'] = GraphTokenDataset
        globals()['add_query_encoding_to_features'] = add_query_encoding_to_features
        globals()['AddQueryEncoding'] = AddQueryEncoding
        globals()['GraphTokenDatasetForAutoGraph'] = GraphTokenDatasetForAutoGraph
        globals()['ZINCTokenizationDataset'] = ZINCTokenizationDataset
        globals()['collate_zinc_batch'] = collate_zinc_batch
        globals()['ZINCDatasetForAutoGraph'] = ZINCDatasetForAutoGraph
        globals()['get_zinc_num_types'] = get_zinc_num_types
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'SPECIAL',
    'load_examples',
    'load_examples_multi_algorithm',
    'determine_num_classes',
    'determine_num_classes_pyg',
    'build_vocab_from_texts',
    'TokenDataset',
    'collate',
    'resolve_split_globs',
    'resolve_multi_algorithm_globs',
    'parse_distance_label_from_text',
    'parse_query_nodes_from_text',
    'balance_classes',
    'get_balanced_indices',
    'GraphTokenDataset',
    'add_query_encoding_to_features',
    'AddQueryEncoding',
    'GraphTokenDatasetForAutoGraph',
    'ZINCTokenizationDataset',
    'collate_zinc_batch',
    'ZINCDatasetForAutoGraph',
    'get_zinc_num_types',
    # ZINC fixed vocabulary
    'build_fixed_zinc_vocab',
    'get_atom_type_id',
    'get_bond_type_id',
    'get_atom_type_from_id',
    'get_bond_type_from_id',
    'extend_vocab_with_dynamic_tokens',
    'map_autograph_token_to_fixed_id',
    'ZINC_ATOM_TYPES',
    'ZINC_BOND_TYPES',
    'NUM_ATOM_TYPES',
    'NUM_BOND_TYPES',
]
