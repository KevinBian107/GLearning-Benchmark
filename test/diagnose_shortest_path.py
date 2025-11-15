"""
Diagnostic script to identify why shortest_path task has poor performance.

Checks:
1. Data quality and label distribution
2. Query encoding presence
3. Model seeing the queries correctly
4. Baseline accuracy (random guessing)
"""

import json
import sys
import os
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_data_loader import load_examples, parse_distance_label_from_text, parse_query_nodes_from_text


def analyze_data_quality(task_dir="graph-token/tasks_train/shortest_path/er/train"):
    """Analyze the shortest_path training data."""
    print("\n" + "="*80)
    print("1. DATA QUALITY ANALYSIS")
    print("="*80)

    # Load some examples
    import glob
    files = glob.glob(f"{task_dir}/*.json")[:100]  # Sample 100 files

    labels = []
    queries = []
    texts = []

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            # Each file contains a list of examples
            if isinstance(data, list):
                for item in data:
                    text = item.get('text', '')
                    label = parse_distance_label_from_text(text)
                    query = parse_query_nodes_from_text(text)

                    if label is not None:
                        labels.append(label)
                        queries.append(query)
                        texts.append(text)
            else:
                # Single example format
                text = data.get('text', '')
                label = parse_distance_label_from_text(text)
                query = parse_query_nodes_from_text(text)

                if label is not None:
                    labels.append(label)
                    queries.append(query)
                    texts.append(text)

    print(f"✓ Loaded {len(labels)} examples from {len(files)} files")

    # Check label distribution
    label_counts = Counter(labels)
    print(f"\n📊 Label Distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = 100 * count / len(labels)
        print(f"  len{label+1} (class {label}): {count:4d} ({pct:5.1f}%)")

    # Check query presence
    queries_present = sum(1 for q in queries if q is not None)
    print(f"\n🔍 Query Encoding:")
    print(f"  Examples with query nodes: {queries_present}/{len(queries)} ({100*queries_present/len(queries):.1f}%)")

    if queries_present > 0:
        print(f"  Sample queries: {queries[:5]}")
    else:
        print("  ⚠️  WARNING: No query nodes found in data!")

    # Show example texts
    print(f"\n📝 Sample Input Texts (first 3):")
    for i, text in enumerate(texts[:3]):
        print(f"\n  Example {i+1}:")
        print(f"  Text (first 100 chars): {text[:100]}...")
        print(f"  Label: len{labels[i]+1} (class {labels[i]})")
        print(f"  Query: {queries[i]}")

    # Calculate baseline accuracy (random guessing)
    most_common_label = label_counts.most_common(1)[0][0]
    majority_baseline = label_counts[most_common_label] / len(labels)
    uniform_baseline = 1.0 / len(label_counts)

    print(f"\n📈 Baseline Accuracies:")
    print(f"  Random guessing (uniform): {uniform_baseline:.2%}")
    print(f"  Majority class baseline:   {majority_baseline:.2%}")

    return labels, queries, texts


def check_query_encoding_in_models():
    """Check how different models encode queries."""
    print("\n" + "="*80)
    print("2. QUERY ENCODING CHECK")
    print("="*80)

    # Check IBTT/AGTT (sequence models)
    print("\n🔤 Sequence Models (IBTT/AGTT):")
    print("  Query is embedded in text: '<q> shortest_distance u v'")
    print("  Models must learn to read this from the sequence")

    # Check MPNN/GPS (graph models)
    print("\n📊 Graph Models (MPNN/GPS):")

    # Check if AddQueryEncoding is used
    import trainer.train_mpnn as train_mpnn_module
    import trainer.train_ggps as train_ggps_module

    # Check train_mpnn.py for query encoding
    with open('trainer/train_mpnn.py', 'r') as f:
        mpnn_code = f.read()
        if 'AddQueryEncoding' in mpnn_code:
            print("  ✓ MPNN uses AddQueryEncoding transform")
        else:
            print("  ⚠️  MPNN may NOT be using AddQueryEncoding!")

    # Check train_ggps.py for query encoding
    with open('trainer/train_ggps.py', 'r') as f:
        ggps_code = f.read()
        if 'AddQueryEncoding' in ggps_code:
            print("  ✓ GGPS uses AddQueryEncoding transform")
        else:
            print("  ⚠️  GGPS may NOT be using AddQueryEncoding!")


def check_config_settings():
    """Check if config settings are appropriate."""
    print("\n" + "="*80)
    print("3. CONFIGURATION ANALYSIS")
    print("="*80)

    import yaml

    configs = {
        'IBTT': 'configs/ibtt_graph_token.yaml',
        'MPNN': 'configs/mpnn_graph_token.yaml',
        'AGTT': 'configs/agtt_graph_token.yaml',
        'GGPS': 'configs/gps_graph_token.yaml',
    }

    for name, path in configs.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            print(f"\n📋 {name} Config:")

            # Check task
            task = config.get('dataset', {}).get('task') or config.get('data', {}).get('task')
            print(f"  Task: {task}")

            # Check num_classes
            model_cfg = config.get('model', {})
            num_classes = model_cfg.get('num_classes')
            print(f"  Num classes: {num_classes}")

            if task == 'shortest_path' and num_classes != 7:
                print(f"  ⚠️  WARNING: shortest_path should have 7 classes, got {num_classes}!")

            # Check data fraction
            data_frac = config.get('dataset', {}).get('data_fraction') or config.get('data', {}).get('data_fraction')
            if data_frac:
                print(f"  Data fraction: {data_frac}")
                if data_frac < 1.0:
                    print(f"  ℹ️  Using only {data_frac*100:.0f}% of data")

            # Check model size
            if 'd_model' in model_cfg:
                print(f"  Model dim: {model_cfg.get('d_model')}")
            if 'hidden_dim' in model_cfg:
                print(f"  Hidden dim: {model_cfg.get('hidden_dim')}")

            # Check learning rate
            train_cfg = config.get('train', {})
            lr = train_cfg.get('lr')
            if lr:
                print(f"  Learning rate: {lr}")


def check_actual_data_loading():
    """Test actual data loading with the data_loader."""
    print("\n" + "="*80)
    print("4. ACTUAL DATA LOADING TEST")
    print("="*80)

    try:
        # Try loading with the actual data loader
        examples = load_examples(
            "graph-token/tasks_train/shortest_path/er/train/*.json",
            task='shortest_path',
            data_fraction=0.01,  # Just 1% for quick test
            seed=42
        )

        print(f"✓ Loaded {len(examples)} examples using data_loader")

        # Check labels
        labels = [ex.get('label') for ex in examples if ex.get('label') is not None]
        queries = [ex.get('query_nodes') for ex in examples]

        print(f"  Examples with labels: {len(labels)}/{len(examples)}")
        print(f"  Label distribution: {Counter(labels)}")
        print(f"  Examples with query_nodes: {sum(1 for q in queries if q is not None)}")

        # Show a sample
        if examples:
            sample = examples[0]
            print(f"\n  Sample example:")
            for key, value in sample.items():
                if key == 'text':
                    print(f"    {key}: {str(value)[:100]}...")
                else:
                    print(f"    {key}: {value}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SHORTEST PATH PERFORMANCE DIAGNOSIS")
    print("="*80)

    try:
        # Run all diagnostics
        labels, queries, texts = analyze_data_quality()
        check_query_encoding_in_models()
        check_config_settings()
        check_actual_data_loading()

        print("\n" + "="*80)
        print("DIAGNOSIS COMPLETE")
        print("="*80)

        print("\n🔍 Key Findings:")
        print("  Check the warnings (⚠️) above for potential issues")
        print("\n💡 Common Issues to Check:")
        print("  1. Is query encoding (u, v) actually being used by models?")
        print("  2. Is the label distribution very imbalanced?")
        print("  3. Are model dimensions too small?")
        print("  4. Is the learning rate appropriate?")
        print("  5. Is data_fraction too small?")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
