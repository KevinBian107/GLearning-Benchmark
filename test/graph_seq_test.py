"""
Test to verify that graph-native and sequence-based methods receive the same data.

This test checks that:
1. Both representations load the same number of samples
2. Labels match between graph and sequence representations
3. Graph structure (nodes/edges) matches between representations
4. Samples are in the same order
"""

import os
import sys
import json
from glob import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from graph_token_dataset import GraphTokenDataset
from data_loader import load_examples, resolve_split_globs


def extract_graph_from_text(text):
    """
    Extract nodes and edges from tokenized text.
    Mirrors the logic in graph_token_dataset.py

    Format: "src dst <e> src dst <e> ..." (edge pairs come BEFORE <e>)
    """
    tokens = text.split()
    edges = []
    nodes = []

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

    # If no explicit nodes, infer from edges
    if len(nodes) == 0 and len(edges) > 0:
        node_set = set()
        for src, tgt in edges:
            node_set.add(src)
            node_set.add(tgt)
        nodes = sorted(list(node_set))

    return nodes, edges


def extract_label_from_text(text):
    """Extract label from tokenized text."""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok == '<p>' and i + 1 < len(tokens):
            label_tok = tokens[i + 1].lower()
            if label_tok == 'yes':
                return 1
            elif label_tok == 'no':
                return 0
    return None


def test_data_consistency(task='cycle_check', algorithm='er', split='train'):
    """
    Test that graph and sequence representations contain the same data.
    """
    print(f"\n{'='*70}")
    print(f"Testing Data Consistency: {task} / {algorithm} / {split}")
    print(f"{'='*70}\n")

    # 1. Load graph-native dataset
    print("Loading graph-native dataset (GraphTokenDataset)...")
    graph_dataset = GraphTokenDataset(
        root='graph-token',
        task=task,
        algorithm=algorithm,
        split=split,
        use_split_tasks_dirs=True,
    )
    print(f"  ✓ Loaded {len(graph_dataset)} graphs")

    # 2. Load sequence dataset
    print("\nLoading sequence dataset (TokenDataset via load_examples)...")
    train_glob, val_glob, test_glob = resolve_split_globs(
        root='graph-token',
        task=task,
        algorithm=algorithm,
        use_split_tasks_dirs=True,
    )

    split_glob = {'train': train_glob, 'val': val_glob, 'test': test_glob}[split]
    seq_examples = load_examples(split_glob)
    print(f"  ✓ Loaded {len(seq_examples)} sequences")

    # 3. Check sample counts match
    print(f"\n{'Test 1: Sample Count':-^70}")
    if len(graph_dataset) == len(seq_examples):
        print(f"  ✓ PASS: Both have {len(graph_dataset)} samples")
    else:
        print(f"  ✗ FAIL: Graph has {len(graph_dataset)}, Sequence has {len(seq_examples)}")
        return False

    # 4. Check labels match
    print(f"\n{'Test 2: Label Consistency':-^70}")
    label_mismatches = []
    for i in range(min(len(graph_dataset), len(seq_examples))):
        graph_label = graph_dataset[i].y.item()
        seq_label = seq_examples[i].get('label')

        if seq_label is None:
            seq_label = extract_label_from_text(seq_examples[i]['text'])

        if graph_label != seq_label:
            label_mismatches.append((i, graph_label, seq_label))

    if len(label_mismatches) == 0:
        print(f"  ✓ PASS: All {len(graph_dataset)} labels match")
    else:
        print(f"  ✗ FAIL: {len(label_mismatches)} label mismatches found:")
        for idx, g_label, s_label in label_mismatches[:5]:
            print(f"    Sample {idx}: graph={g_label}, sequence={s_label}")
        return False

    # 5. Check graph structure matches
    print(f"\n{'Test 3: Graph Structure Consistency':-^70}")
    structure_mismatches = []

    for i in range(min(10, len(graph_dataset))):  # Check first 10 samples
        # Get graph data
        graph_data = graph_dataset[i]
        graph_nodes = list(range(graph_data.num_nodes))
        graph_edges = graph_data.edge_index.t().tolist()
        graph_edges = [(e[0], e[1]) for e in graph_edges]
        graph_edges_set = set(graph_edges)

        # Get sequence data
        seq_text = seq_examples[i]['text']
        seq_nodes, seq_edges = extract_graph_from_text(seq_text)
        seq_edges_set = set(seq_edges)

        # Compare
        if graph_edges_set != seq_edges_set:
            structure_mismatches.append({
                'index': i,
                'graph_edges': len(graph_edges_set),
                'seq_edges': len(seq_edges_set),
                'only_in_graph': graph_edges_set - seq_edges_set,
                'only_in_seq': seq_edges_set - graph_edges_set,
            })

    if len(structure_mismatches) == 0:
        print(f"  ✓ PASS: All checked structures match")
    else:
        print(f"  ✗ FAIL: {len(structure_mismatches)} structure mismatches found:")
        for mismatch in structure_mismatches[:3]:
            print(f"    Sample {mismatch['index']}:")
            print(f"      Graph edges: {mismatch['graph_edges']}, Seq edges: {mismatch['seq_edges']}")
            if mismatch['only_in_graph']:
                print(f"      Only in graph: {list(mismatch['only_in_graph'])[:5]}")
            if mismatch['only_in_seq']:
                print(f"      Only in seq: {list(mismatch['only_in_seq'])[:5]}")
        return False

    # 6. Label distribution check
    print(f"\n{'Test 4: Label Distribution':-^70}")
    graph_labels = [graph_dataset[i].y.item() for i in range(len(graph_dataset))]
    seq_labels = [seq_examples[i].get('label') or extract_label_from_text(seq_examples[i]['text'])
                  for i in range(len(seq_examples))]

    graph_pos = sum(graph_labels)
    seq_pos = sum(seq_labels)

    print(f"  Graph dataset: {graph_pos}/{len(graph_labels)} positive ({100*graph_pos/len(graph_labels):.1f}%)")
    print(f"  Seq dataset:   {seq_pos}/{len(seq_labels)} positive ({100*seq_pos/len(seq_labels):.1f}%)")

    if graph_pos == seq_pos:
        print(f"  ✓ PASS: Same label distribution")
    else:
        print(f"  ✗ FAIL: Different label distributions")
        return False

    # 7. Sample a few examples for manual inspection
    print(f"\n{'Sample Inspection':-^70}")
    for i in [0, len(graph_dataset)//2, len(graph_dataset)-1]:
        graph_data = graph_dataset[i]
        seq_text = seq_examples[i]['text']

        print(f"\nSample {i}:")
        print(f"  Label: {graph_data.y.item()}")
        print(f"  Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.size(1)} edges")
        print(f"  Sequence (first 100 chars): {seq_text[:100]}...")

    print(f"\n{'='*70}")
    print("  ✓ ALL TESTS PASSED")
    print(f"{'='*70}\n")

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test data consistency between graph and sequence representations')
    parser.add_argument('--task', type=str, default='cycle_check',
                        help='Task name')
    parser.add_argument('--algorithm', type=str, default='er',
                        help='Graph generation algorithm')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Data split to test')

    args = parser.parse_args()

    try:
        success = test_data_consistency(
            task=args.task,
            algorithm=args.algorithm,
            split=args.split
        )

        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
