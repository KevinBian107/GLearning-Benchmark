"""
Test to analyze training performance and understand why accuracy starts high.

This test checks:
1. Label distribution (is the dataset balanced?)
2. Baseline accuracy (random guessing, majority class prediction)
3. Data leakage (duplicates between splits, train/val/test overlap)
4. Task difficulty (manual inspection, graph properties)
5. Model sanity checks (untrained model performance)
"""

import os
import sys
from collections import Counter
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from graph_token_dataset import GraphTokenDataset
from train_mpnn import MPNN


def has_cycle_dfs(num_nodes, edge_list):
    """
    Check if an UNDIRECTED graph has a cycle using DFS.
    Returns True if cycle exists, False otherwise.

    Note: This implementation is for undirected graphs where each edge
    is bidirectional. We need to track the parent node to avoid counting
    the edge we came from as a cycle.
    """
    if len(edge_list) == 0:
        return False

    # Build adjacency list for UNDIRECTED graph
    adj = [[] for _ in range(num_nodes)]
    for src, tgt in edge_list:
        adj[src].append(tgt)
        adj[tgt].append(src)  # Add both directions for undirected graph

    visited = [False] * num_nodes

    def dfs(node, parent):
        """DFS with parent tracking for undirected graphs."""
        visited[node] = True

        for neighbor in adj[node]:
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                # Found a cycle: visiting an already-visited node that's not our parent
                return True

        return False

    # Check all connected components
    for node in range(num_nodes):
        if not visited[node]:
            if dfs(node, -1):
                return True

    return False


def compute_graph_hash(graph_data):
    """Compute a hash for a graph to detect duplicates."""
    edges = graph_data.edge_index.t().tolist()
    edges_sorted = tuple(sorted(tuple(sorted(e)) for e in edges))
    return (graph_data.num_nodes, edges_sorted, graph_data.y.item())


def test_label_distribution(task='cycle_check', algorithm='er'):
    """Test 1: Check label distribution across splits."""
    print(f"\n{'='*70}")
    print(f"Test 1: Label Distribution Analysis")
    print(f"{'='*70}\n")

    results = {}
    for split in ['train', 'val', 'test']:
        try:
            dataset = GraphTokenDataset(
                root='graph-token',
                task=task,
                algorithm=algorithm,
                split=split,
                use_split_tasks_dirs=True,
            )
            labels = [dataset[i].y.item() for i in range(len(dataset))]
            label_counts = Counter(labels)

            results[split] = {
                'total': len(labels),
                'positive': label_counts.get(1, 0),
                'negative': label_counts.get(0, 0),
            }

            pos_pct = 100 * results[split]['positive'] / results[split]['total']
            print(f"{split.upper():5s}: {results[split]['total']:4d} samples | "
                  f"{results[split]['positive']:4d} positive ({pos_pct:.1f}%) | "
                  f"{results[split]['negative']:4d} negative ({100-pos_pct:.1f}%)")

        except Exception as e:
            print(f"{split.upper():5s}: Not available ({e})")

    # Calculate baseline accuracies
    print(f"\n{'Baseline Accuracies':-^70}")
    for split, data in results.items():
        total = data['total']
        pos = data['positive']
        neg = data['negative']

        # Random guessing (50/50)
        random_acc = 50.0

        # Majority class prediction
        majority_acc = 100 * max(pos, neg) / total

        # Random with true distribution
        dist_random_acc = 100 * ((pos/total)**2 + (neg/total)**2)

        print(f"\n{split.upper()}:")
        print(f"  Random guessing (50/50):        {random_acc:.1f}%")
        print(f"  Majority class prediction:      {majority_acc:.1f}%")
        print(f"  Random (matching distribution): {dist_random_acc:.1f}%")

    # Check if distribution is too skewed
    print(f"\n{'Assessment':-^70}")
    for split, data in results.items():
        pos_pct = 100 * data['positive'] / data['total']
        if 40 <= pos_pct <= 60:
            print(f"  ✓ {split.upper()}: Balanced ({pos_pct:.1f}% positive)")
        elif 30 <= pos_pct <= 70:
            print(f"  ⚠ {split.upper()}: Slightly imbalanced ({pos_pct:.1f}% positive)")
        else:
            print(f"  ✗ {split.upper()}: Heavily imbalanced ({pos_pct:.1f}% positive)")
            print(f"      → A model predicting majority class gets {max(pos_pct, 100-pos_pct):.1f}% accuracy!")

    return results


def test_data_leakage(task='cycle_check', algorithm='er'):
    """Test 2: Check for data leakage between splits."""
    print(f"\n{'='*70}")
    print(f"Test 2: Data Leakage Analysis")
    print(f"{'='*70}\n")

    # Load all splits
    datasets = {}
    for split in ['train', 'val', 'test']:
        try:
            datasets[split] = GraphTokenDataset(
                root='graph-token',
                task=task,
                algorithm=algorithm,
                split=split,
                use_split_tasks_dirs=True,
            )
            print(f"Loaded {split}: {len(datasets[split])} samples")
        except Exception as e:
            print(f"Could not load {split}: {e}")

    if len(datasets) < 2:
        print("  ⚠ Not enough splits to check for leakage")
        return

    # Compute hashes for all graphs
    print(f"\n{'Computing graph hashes...':-^70}")
    hashes = {}
    for split, dataset in datasets.items():
        hashes[split] = set()
        for i in range(len(dataset)):
            graph_hash = compute_graph_hash(dataset[i])
            hashes[split].add(graph_hash)
        print(f"  {split}: {len(hashes[split])} unique graphs (from {len(dataset)} total)")

    # Check for overlaps
    print(f"\n{'Checking for overlaps...':-^70}")
    pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
    has_leakage = False

    for split1, split2 in pairs:
        if split1 not in hashes or split2 not in hashes:
            continue

        overlap = hashes[split1] & hashes[split2]
        if len(overlap) > 0:
            print(f"  ✗ {split1.upper()} ∩ {split2.upper()}: {len(overlap)} duplicate graphs found!")
            has_leakage = True

            # Show a sample
            sample = list(overlap)[0]
            print(f"      Example: {sample[0]} nodes, {len(sample[1])} edges, label={sample[2]}")
        else:
            print(f"  ✓ {split1.upper()} ∩ {split2.upper()}: No duplicates")

    if has_leakage:
        print(f"\n  ⚠ DATA LEAKAGE DETECTED! This explains high initial accuracy.")
        print(f"     Models may have seen test data during training!")
    else:
        print(f"\n  ✓ No data leakage detected")

    return has_leakage


def test_task_difficulty(task='cycle_check', algorithm='er', num_samples=10):
    """Test 3: Manually check task difficulty."""
    print(f"\n{'='*70}")
    print(f"Test 3: Task Difficulty Assessment")
    print(f"{'='*70}\n")

    dataset = GraphTokenDataset(
        root='graph-token',
        task=task,
        algorithm=algorithm,
        split='train',
        use_split_tasks_dirs=True,
    )

    print(f"Manually checking {num_samples} random samples...\n")

    # Sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    correct_predictions = 0
    for idx in indices:
        graph = dataset[idx]
        num_nodes = graph.num_nodes
        edges = graph.edge_index.t().tolist()
        true_label = graph.y.item()

        # For cycle detection, we can verify manually
        if task == 'cycle_check':
            predicted_label = 1 if has_cycle_dfs(num_nodes, edges) else 0

            match = "✓" if predicted_label == true_label else "✗"
            correct_predictions += (predicted_label == true_label)

            print(f"Sample {idx}: {num_nodes} nodes, {len(edges)} edges")
            print(f"  True label: {true_label} ({'has cycle' if true_label else 'no cycle'})")
            print(f"  DFS check:  {predicted_label} ({'has cycle' if predicted_label else 'no cycle'}) {match}")
            print()

    if task == 'cycle_check':
        manual_acc = 100 * correct_predictions / len(indices)
        print(f"{'Manual Verification':-^70}")
        print(f"  Correct: {correct_predictions}/{len(indices)} ({manual_acc:.1f}%)")

        if manual_acc < 100:
            print(f"  ✗ Labels may be incorrect or DFS implementation differs!")
        else:
            print(f"  ✓ Labels are correct (verified with DFS)")

        # Task difficulty assessment
        print(f"\n{'Task Difficulty':-^70}")
        # Count how many graphs have cycles
        num_with_cycles = sum(1 for i in indices if dataset[i].y.item() == 1)
        print(f"  {num_with_cycles}/{len(indices)} sampled graphs have cycles")
        print(f"  Cycle detection is algorithmically simple (DFS traversal)")
        print(f"  → Neural networks should learn this easily")


def test_untrained_model_performance(task='cycle_check', algorithm='er'):
    """Test 4: Check untrained model performance."""
    print(f"\n{'='*70}")
    print(f"Test 4: Untrained Model Performance")
    print(f"{'='*70}\n")

    # Load dataset
    dataset = GraphTokenDataset(
        root='graph-token',
        task=task,
        algorithm=algorithm,
        split='train',
        use_split_tasks_dirs=True,
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Create untrained model
    model = MPNN(
        in_dim=1,
        hidden_dim=64,
        num_layers=3,
        dropout=0.0,
        pooling='mean',
    )
    model.eval()

    print("Testing untrained MPNN model (random weights)...")

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            preds = logits.argmax(dim=-1).tolist()
            labels = batch.y.squeeze().tolist()

            if isinstance(labels, int):
                labels = [labels]
            if isinstance(preds, int):
                preds = [preds]

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = 100 * correct / total

    print(f"\n  Untrained model accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Count prediction distribution
    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)

    print(f"\n  Prediction distribution: {dict(pred_counts)}")
    print(f"  True label distribution: {dict(label_counts)}")

    print(f"\n{'Assessment':-^70}")
    if 40 <= accuracy <= 60:
        print(f"  ✓ Untrained model performs close to random (expected)")
    elif accuracy > 80:
        print(f"  ✗ Untrained model has suspiciously high accuracy!")
        print(f"     Possible issues:")
        print(f"     1. Model is predicting majority class")
        print(f"     2. Task is too easy")
        print(f"     3. Data is leaking")
    else:
        print(f"  ⚠ Untrained model performance is {accuracy:.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze training performance')
    parser.add_argument('--task', type=str, default='cycle_check',
                        help='Task name')
    parser.add_argument('--algorithm', type=str, default='er',
                        help='Graph generation algorithm')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples for manual inspection')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Training Performance Analysis")
    print(f"Task: {args.task} | Algorithm: {args.algorithm}")
    print(f"{'='*70}")

    # Run all tests
    results = test_label_distribution(args.task, args.algorithm)
    leakage = test_data_leakage(args.task, args.algorithm)
    test_task_difficulty(args.task, args.algorithm, args.samples)
    test_untrained_model_performance(args.task, args.algorithm)

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
