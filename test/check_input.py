"""
Check input formats and prediction types across all tasks.

This script analyzes:
1. Tokenized sequence format (for GTT/Transformer)
2. Native graph format (for MPNN/GPS)
3. Prediction types (binary classification, regression, etc.)
4. Label distributions and statistics
"""

import os
import sys
import json
from glob import glob
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_token_dataset import parse_graph_from_text, parse_label_from_text


def analyze_task(task: str, algorithm: str = 'er', max_samples: int = 100) -> Dict[str, Any]:
    """Analyze a specific task's input format and prediction type."""

    print(f"\n{'='*80}")
    print(f"Task: {task} | Algorithm: {algorithm}")
    print(f"{'='*80}\n")

    # Try to find data files
    patterns = [
        f"graph-token/tasks_train/{task}/{algorithm}/train/*.json",
        f"graph-token/tasks/{task}/{algorithm}/train/*.json",
    ]

    files = []
    for pattern in patterns:
        files = glob(pattern)
        if files:
            break

    if not files:
        print(f"  ⚠ No data found for task '{task}' with algorithm '{algorithm}'")
        return {}

    print(f"Found {len(files)} files. Analyzing {min(max_samples, len(files))} samples...\n")

    # Analyze samples
    samples = []
    labels = []
    seq_lengths = []
    num_nodes_list = []
    num_edges_list = []

    for fpath in files[:max_samples]:
        with open(fpath) as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, list):
            data = data[0] if data else {}

        if not isinstance(data, dict):
            continue

        text = data.get('text', '')
        if not text:
            continue

        # Parse tokenized format
        tokens = text.split()
        seq_lengths.append(len(tokens))

        # Parse graph structure
        nodes, edges = parse_graph_from_text(text)
        num_nodes = len(nodes) if nodes else (max([max(e) for e in edges]) + 1 if edges else 0)
        num_nodes_list.append(num_nodes)
        num_edges_list.append(len(edges))

        # Parse label
        label = data.get('label')
        if label is None:
            label = parse_label_from_text(text)

        # Try to extract numeric labels for regression tasks
        if label is None:
            # Look for numeric values after <p>
            for i, tok in enumerate(tokens):
                if tok == '<p>' and i + 1 < len(tokens):
                    try:
                        label = float(tokens[i + 1])
                    except ValueError:
                        pass

        labels.append(label)
        samples.append({
            'text': text,
            'tokens': tokens,
            'nodes': nodes,
            'edges': edges,
            'label': label,
            'num_nodes': num_nodes,
            'num_edges': len(edges),
        })

    if not samples:
        print("  ⚠ No valid samples found")
        return {}

    # Determine prediction type
    prediction_type = determine_prediction_type(labels)

    # Print format analysis
    print(f"{'Tokenized Sequence Format':-^80}")
    print_tokenized_format(samples[:3])

    print(f"\n{'Native Graph Format':-^80}")
    print_native_graph_format(samples[:3])

    print(f"\n{'Prediction Type Analysis':-^80}")
    print_prediction_type(labels, prediction_type)

    print(f"\n{'Dataset Statistics':-^80}")
    print_statistics(seq_lengths, num_nodes_list, num_edges_list, labels)

    return {
        'task': task,
        'algorithm': algorithm,
        'num_samples': len(samples),
        'prediction_type': prediction_type,
        'samples': samples,
    }


def determine_prediction_type(labels: List[Any]) -> str:
    """Determine if task is binary classification, multi-class, or regression."""

    # Remove None labels
    valid_labels = [l for l in labels if l is not None]

    if not valid_labels:
        return "unknown"

    # Check if all labels are numeric
    try:
        numeric_labels = [float(l) for l in valid_labels]
    except (ValueError, TypeError):
        return "unknown"

    # Check if all labels are 0 or 1 (binary)
    unique_labels = set(numeric_labels)

    if unique_labels.issubset({0, 1, 0.0, 1.0}):
        return "binary_classification"

    # Check if all labels are integers (multi-class)
    if all(l == int(l) for l in numeric_labels):
        if len(unique_labels) <= 10:
            return f"multi_class_classification ({len(unique_labels)} classes)"
        else:
            return "regression (integer-valued)"

    # Otherwise, it's regression
    return "regression (continuous)"


def print_tokenized_format(samples: List[Dict]) -> None:
    """Print tokenized sequence format with annotation."""

    for i, sample in enumerate(samples[:2], 1):
        text = sample['text']
        tokens = sample['tokens']

        print(f"\nExample {i}:")
        print(f"  Full sequence ({len(tokens)} tokens):")

        # Break down the sequence into parts
        parts = []
        if '<bos>' in tokens:
            bos_idx = tokens.index('<bos>')
            parts.append(('BOS', tokens[bos_idx:bos_idx+1]))

        # Find edges (before <n>)
        edge_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == '<n>' or tokens[i] == '<q>':
                break
            if tokens[i] not in ['<bos>', '<eos>']:
                edge_tokens.append(tokens[i])
            i += 1

        if edge_tokens:
            parts.append(('EDGES', edge_tokens[:15] + (['...'] if len(edge_tokens) > 15 else [])))

        # Find nodes
        if '<n>' in tokens:
            n_idx = tokens.index('<n>')
            node_tokens = []
            i = n_idx + 1
            while i < len(tokens) and tokens[i] not in ['<q>', '<p>', '<eos>']:
                node_tokens.append(tokens[i])
                i += 1
            parts.append(('NODES', [tokens[n_idx]] + node_tokens[:10] + (['...'] if len(node_tokens) > 10 else [])))

        # Find query
        if '<q>' in tokens:
            q_idx = tokens.index('<q>')
            query_tokens = []
            i = q_idx + 1
            while i < len(tokens) and tokens[i] not in ['<p>', '<eos>']:
                query_tokens.append(tokens[i])
                i += 1
            parts.append(('QUERY', [tokens[q_idx]] + query_tokens))

        # Find answer
        if '<p>' in tokens:
            p_idx = tokens.index('<p>')
            answer_tokens = []
            i = p_idx + 1
            while i < len(tokens) and tokens[i] != '<eos>':
                answer_tokens.append(tokens[i])
                i += 1
            parts.append(('ANSWER', [tokens[p_idx]] + answer_tokens))

        if '<eos>' in tokens:
            parts.append(('EOS', ['<eos>']))

        # Print formatted
        for part_name, part_tokens in parts:
            print(f"    {part_name:10s}: {' '.join(map(str, part_tokens))}")


def print_native_graph_format(samples: List[Dict]) -> None:
    """Print native graph format (PyG Data object structure)."""

    for i, sample in enumerate(samples[:2], 1):
        print(f"\nExample {i}:")
        print(f"  Graph structure:")
        print(f"    num_nodes: {sample['num_nodes']}")
        print(f"    num_edges: {sample['num_edges']}")

        if sample['edges']:
            print(f"    edge_index (first 10): {sample['edges'][:10]}")
        else:
            print(f"    edge_index: [] (no edges)")

        if sample['nodes']:
            print(f"    nodes: {sample['nodes'][:15]}{' ...' if len(sample['nodes']) > 15 else ''}")

        print(f"    label: {sample['label']}")

        # Show PyG Data structure
        print(f"\n  PyG Data object format:")
        print(f"    Data(")
        print(f"      x=Tensor[{sample['num_nodes']}, 1],          # Node features (constant 1s)")
        print(f"      edge_index=Tensor[2, {sample['num_edges']}],  # Edge list in COO format")
        print(f"      y=Tensor[1],                    # Label")
        print(f"      num_nodes={sample['num_nodes']}")
        print(f"    )")


def print_prediction_type(labels: List[Any], pred_type: str) -> None:
    """Print prediction type and label statistics."""

    valid_labels = [l for l in labels if l is not None]

    print(f"\n  Prediction Type: {pred_type}")
    print(f"  Total samples: {len(labels)}")
    print(f"  Valid labels: {len(valid_labels)}")
    print(f"  Missing labels: {len(labels) - len(valid_labels)}")

    if not valid_labels:
        return

    # Convert to numeric
    try:
        numeric_labels = [float(l) for l in valid_labels]
    except (ValueError, TypeError):
        print(f"  Label types: {set(type(l).__name__ for l in valid_labels)}")
        print(f"  Unique labels: {set(valid_labels)}")
        return

    print(f"\n  Label Statistics:")

    if 'binary' in pred_type or 'multi_class' in pred_type:
        # Classification: show class distribution
        label_counts = Counter(numeric_labels)
        total = len(numeric_labels)

        print(f"    Class distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = 100 * count / total
            label_name = {0: 'negative', 1: 'positive'}.get(int(label), f'class_{int(label)}')
            print(f"      {int(label):3d} ({label_name:10s}): {count:4d} ({pct:5.1f}%)")

        # Check balance
        if len(label_counts) == 2:
            minority_pct = min(label_counts.values()) / total * 100
            if minority_pct < 30:
                print(f"    ⚠ Imbalanced: {minority_pct:.1f}% minority class")
            elif 40 <= minority_pct <= 60:
                print(f"    ✓ Balanced: {minority_pct:.1f}% / {100-minority_pct:.1f}%")

    else:
        # Regression: show numeric statistics
        print(f"    Min:    {min(numeric_labels):.2f}")
        print(f"    Max:    {max(numeric_labels):.2f}")
        print(f"    Mean:   {sum(numeric_labels)/len(numeric_labels):.2f}")
        print(f"    Median: {sorted(numeric_labels)[len(numeric_labels)//2]:.2f}")

        # Show distribution
        print(f"    Unique values: {len(set(numeric_labels))}")


def print_statistics(seq_lengths: List[int], num_nodes: List[int],
                     num_edges: List[int], labels: List[Any]) -> None:
    """Print dataset statistics."""

    def stats(values, name):
        if not values:
            return
        print(f"  {name}:")
        print(f"    Min: {min(values)}, Max: {max(values)}, "
              f"Mean: {sum(values)/len(values):.1f}, "
              f"Median: {sorted(values)[len(values)//2]}")

    stats(seq_lengths, "Sequence length (tokens)")
    stats(num_nodes, "Number of nodes")
    stats(num_edges, "Number of edges")


def main():
    """Check all available tasks."""

    print(f"\n{'='*80}")
    print(f"Graph-Token Input Format Checker")
    print(f"{'='*80}")

    # List of tasks to check
    tasks = [
        'cycle_check',
        'connected_nodes',
        'edge_existence',
        'node_degree',
        'node_count',
        'edge_count',
    ]

    algorithms = ['er', 'ba', 'sbm', 'ws', 'grid']

    results = {}

    for task in tasks:
        # Try first available algorithm
        for algo in algorithms:
            result = analyze_task(task, algo, max_samples=50)
            if result:
                results[task] = result
                break

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}\n")

    if not results:
        print("  ⚠ No tasks found. Make sure graph-token data is generated.")
        print("  Run: cd graph-token && bash generate.sh")
        return

    print(f"Found {len(results)} tasks:\n")

    print(f"{'Task':<20} {'Samples':<10} {'Prediction Type':<30}")
    print(f"{'-'*20} {'-'*10} {'-'*30}")
    for task, info in results.items():
        print(f"{task:<20} {info['num_samples']:<10} {info['prediction_type']:<30}")

    print(f"\n{'Task Type Breakdown':-^80}")

    binary_tasks = [t for t, i in results.items() if 'binary' in i['prediction_type']]
    regression_tasks = [t for t, i in results.items() if 'regression' in i['prediction_type']]
    multiclass_tasks = [t for t, i in results.items() if 'multi_class' in i['prediction_type']]

    if binary_tasks:
        print(f"\n  Binary Classification Tasks:")
        for task in binary_tasks:
            print(f"    - {task}")
            print(f"      Output: 2 classes (0/1 or negative/positive)")
            print(f"      Loss: CrossEntropyLoss or BCEWithLogitsLoss")
            print(f"      Metric: Accuracy, F1-score, AUC")

    if multiclass_tasks:
        print(f"\n  Multi-Class Classification Tasks:")
        for task in multiclass_tasks:
            num_classes = results[task]['prediction_type'].split('(')[1].split()[0]
            print(f"    - {task} ({num_classes} classes)")
            print(f"      Output: {num_classes} classes")
            print(f"      Loss: CrossEntropyLoss")
            print(f"      Metric: Accuracy, macro/micro F1")

    if regression_tasks:
        print(f"\n  Regression Tasks:")
        for task in regression_tasks:
            print(f"    - {task}")
            print(f"      Output: Continuous value")
            print(f"      Loss: MSELoss or L1Loss")
            print(f"      Metric: MAE, RMSE, R²")

    print(f"\n{'Model Configuration Notes':-^80}")
    print("""
  For Binary Classification (e.g., cycle_check):
    - GTT: model outputs 2 logits → CrossEntropyLoss
    - MPNN: classifier = nn.Linear(hidden_dim, 2)
    - GPS: dim_out = 1, use BCEWithLogitsLoss

  For Regression (e.g., node_degree, node_count):
    - GTT: model outputs 1 value → MSELoss
    - MPNN: classifier = nn.Linear(hidden_dim, 1)
    - GPS: dim_out = 1, use MSELoss
    - Remove softmax/sigmoid activation

  Important: Check your task type and adjust:
    1. Output dimension (2 for binary, 1 for regression)
    2. Loss function (CrossEntropy vs MSE)
    3. Evaluation metrics
    """)

    print(f"{'='*80}\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
