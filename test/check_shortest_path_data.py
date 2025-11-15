"""
Script to check shortest_path data quality, distribution, and statistics.

Usage:
    python test/check_shortest_path_data.py
    python test/check_shortest_path_data.py --algorithm ba
    python test/check_shortest_path_data.py --algorithm er --max_files 500
"""

import json
import glob
import argparse
from collections import Counter, defaultdict
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_data_loader import parse_distance_label_from_text, parse_query_nodes_from_text


def analyze_split(split_name, files, max_files=None):
    """Analyze a single data split (train/val/test)."""
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} SPLIT ANALYSIS")
    print('='*80)

    if max_files:
        files = files[:max_files]

    labels = []
    query_nodes = []
    texts = []
    graph_sizes = []
    edge_counts = []
    unreachable_count = 0

    print(f"Loading {len(files)} files...")

    for f in files:
        with open(f) as fp:
            data = json.load(fp)

            # Each file contains a list of examples (different query pairs on same graph)
            if isinstance(data, list):
                for item in data:
                    text = item.get('text', '')
                    label = parse_distance_label_from_text(text)
                    query = parse_query_nodes_from_text(text)

                    # Parse graph structure info
                    tokens = text.split()

                    # Count nodes (after <n> tag)
                    if '<n>' in tokens:
                        n_idx = tokens.index('<n>')
                        # Find next special token after <n>
                        next_special_idx = len(tokens)
                        for i, t in enumerate(tokens[n_idx+1:], start=n_idx+1):
                            if t.startswith('<'):
                                next_special_idx = i
                                break
                        node_tokens = tokens[n_idx+1:next_special_idx]
                        num_nodes = len(node_tokens)
                        graph_sizes.append(num_nodes)

                    # Count edges (count <e> tags)
                    num_edges = tokens.count('<e>')
                    edge_counts.append(num_edges)

                    if label is not None:
                        labels.append(label)
                        query_nodes.append(query)
                        texts.append(text)
                    else:
                        # Unreachable pair (INF distance)
                        unreachable_count += 1

    print(f"✓ Loaded {len(labels)} reachable examples")
    if unreachable_count > 0:
        print(f"  (Skipped {unreachable_count} unreachable pairs)")

    # Label Distribution
    print(f"\n📊 LABEL DISTRIBUTION")
    print("-" * 80)
    label_counts = Counter(labels)
    total = len(labels)

    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  len{label+1} (class {label}): {count:6d} ({pct:5.1f}%)  {bar}")

    # Baseline accuracies
    print(f"\n📈 BASELINE ACCURACIES")
    print("-" * 80)
    uniform_acc = 1.0 / len(label_counts) if label_counts else 0
    majority_class = label_counts.most_common(1)[0] if label_counts else (0, 0)
    majority_acc = majority_class[1] / total if total > 0 else 0

    print(f"  Random guessing (uniform): {uniform_acc:6.2%}")
    print(f"  Majority class (len{majority_class[0]+1}): {majority_acc:6.2%}")

    # Class imbalance metrics
    if len(label_counts) > 0:
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"\n⚖️  CLASS IMBALANCE")
        print("-" * 80)
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.1f}:1")

        if imbalance_ratio > 100:
            print(f"  ⚠️  SEVERE imbalance! Rare classes have {min_count} examples vs {max_count}")
        elif imbalance_ratio > 10:
            print(f"  ⚠️  Moderate imbalance detected")
        else:
            print(f"  ✓ Relatively balanced")

    # Query node statistics
    print(f"\n🔍 QUERY STATISTICS")
    print("-" * 80)
    valid_queries = [q for q in query_nodes if q is not None]
    print(f"  Examples with query nodes: {len(valid_queries)}/{len(query_nodes)} ({100*len(valid_queries)/max(len(query_nodes),1):.1f}%)")

    if valid_queries:
        query_u_values = [q[0] for q in valid_queries]
        query_v_values = [q[1] for q in valid_queries]

        print(f"  Query source nodes (u): min={min(query_u_values)}, max={max(query_u_values)}, mean={np.mean(query_u_values):.1f}")
        print(f"  Query target nodes (v): min={min(query_v_values)}, max={max(query_v_values)}, mean={np.mean(query_v_values):.1f}")

    # Graph structure statistics
    if graph_sizes:
        print(f"\n📐 GRAPH STRUCTURE")
        print("-" * 80)
        print(f"  Number of nodes: min={min(graph_sizes)}, max={max(graph_sizes)}, mean={np.mean(graph_sizes):.1f}, median={np.median(graph_sizes):.0f}")
        print(f"  Number of edges: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}, median={np.median(edge_counts):.0f}")

        if edge_counts and graph_sizes:
            # Average degree (undirected graphs)
            avg_degrees = [2 * e / n if n > 0 else 0 for e, n in zip(edge_counts, graph_sizes)]
            print(f"  Average degree: mean={np.mean(avg_degrees):.2f}, median={np.median(avg_degrees):.2f}")

    # Sample examples
    print(f"\n📝 SAMPLE EXAMPLES (first 3)")
    print("-" * 80)
    for i, (text, label, query) in enumerate(zip(texts[:3], labels[:3], query_nodes[:3])):
        print(f"\nExample {i+1}:")
        print(f"  Label: len{label+1} (class {label})")
        print(f"  Query: {query}")
        print(f"  Text (first 120 chars): {text[:120]}...")

    return {
        'num_examples': len(labels),
        'label_counts': label_counts,
        'unreachable': unreachable_count,
        'graph_sizes': graph_sizes,
        'edge_counts': edge_counts,
    }


def main():
    parser = argparse.ArgumentParser(description='Check shortest_path data quality and distribution')
    parser.add_argument('--algorithm', type=str, default='er',
                        help='Graph algorithm (er, ba, sbm, etc.)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to analyze per split (for speed)')
    parser.add_argument('--graph_token_root', type=str, default='graph-token',
                        help='Path to graph-token root directory')

    args = parser.parse_args()

    print("="*80)
    print("SHORTEST PATH DATA CHECK")
    print("="*80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Root: {args.graph_token_root}")
    if args.max_files:
        print(f"Max files per split: {args.max_files}")

    # Check if using split directories
    train_path = f"{args.graph_token_root}/tasks_train/shortest_path/{args.algorithm}/train/*.json"
    test_path = f"{args.graph_token_root}/tasks_test/shortest_path/{args.algorithm}/test/*.json"

    train_files = sorted(glob.glob(train_path))
    test_files = sorted(glob.glob(test_path))

    # Also check for val split
    val_path_split = f"{args.graph_token_root}/tasks_train/shortest_path/{args.algorithm}/val/*.json"
    val_files = sorted(glob.glob(val_path_split))

    if not train_files:
        print(f"\n❌ ERROR: No training files found at {train_path}")
        print("Make sure you've generated the data first!")
        return

    print(f"\nFound files:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")

    # Analyze each split
    results = {}

    if train_files:
        results['train'] = analyze_split('train', train_files, args.max_files)

    if val_files:
        results['val'] = analyze_split('val', val_files, args.max_files)

    if test_files:
        results['test'] = analyze_split('test', test_files, args.max_files)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print('='*80)

    print(f"\n{'Split':<10} {'Examples':<12} {'Unreachable':<13} {'Avg Nodes':<12} {'Avg Edges':<12}")
    print("-" * 80)

    for split_name, data in results.items():
        avg_nodes = np.mean(data['graph_sizes']) if data['graph_sizes'] else 0
        avg_edges = np.mean(data['edge_counts']) if data['edge_counts'] else 0
        print(f"{split_name:<10} {data['num_examples']:<12} {data['unreachable']:<13} {avg_nodes:<12.1f} {avg_edges:<12.1f}")

    # Check for distribution shift
    if 'train' in results and 'test' in results:
        print(f"\n🔄 TRAIN vs TEST DISTRIBUTION")
        print("-" * 80)

        train_dist = results['train']['label_counts']
        test_dist = results['test']['label_counts']

        # Check if distributions are similar
        all_labels = sorted(set(train_dist.keys()) | set(test_dist.keys()))

        print(f"{'Class':<10} {'Train %':<12} {'Test %':<12} {'Difference':<12}")
        print("-" * 80)

        max_diff = 0
        for label in all_labels:
            train_pct = 100 * train_dist.get(label, 0) / max(sum(train_dist.values()), 1)
            test_pct = 100 * test_dist.get(label, 0) / max(sum(test_dist.values()), 1)
            diff = abs(train_pct - test_pct)
            max_diff = max(max_diff, diff)

            marker = "⚠️" if diff > 5 else ""
            print(f"len{label+1:<7} {train_pct:<12.1f} {test_pct:<12.1f} {diff:<12.1f} {marker}")

        print("-" * 80)
        if max_diff > 10:
            print("⚠️  Large distribution shift detected between train and test!")
        elif max_diff > 5:
            print("⚠️  Moderate distribution shift detected")
        else:
            print("✓ Train and test distributions are similar")

    print(f"\n{'='*80}")
    print("✅ DATA CHECK COMPLETE")
    print('='*80)

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")

    if 'train' in results:
        train_label_counts = results['train']['label_counts']
        if train_label_counts:
            max_count = max(train_label_counts.values())
            min_count = min(train_label_counts.values())
            imbalance = max_count / min_count if min_count > 0 else float('inf')

            if imbalance > 100:
                print("  • SEVERE class imbalance detected!")
                print("    Consider: data augmentation, class balancing, or focal loss")
            elif imbalance > 10:
                print("  • Moderate class imbalance detected")
                print("    Monitor per-class metrics (macro-F1, per-class recall)")

            # Check if rare classes have enough examples
            rare_classes = [label for label, count in train_label_counts.items() if count < 50]
            if rare_classes:
                print(f"  • Classes with <50 examples: {[f'len{c+1}' for c in rare_classes]}")
                print("    These may be hard to learn - monitor their recall carefully")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
