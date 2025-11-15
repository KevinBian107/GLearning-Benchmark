"""
Test script to demonstrate class balancing feature.

Usage:
    python test/test_class_balancing.py
    python test/test_class_balancing.py --strategy median
"""

import argparse
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_data_loader import load_examples, balance_classes


def main():
    parser = argparse.ArgumentParser(description='Test class balancing on shortest_path data')
    parser.add_argument('--algorithm', type=str, default='er', help='Graph algorithm')
    parser.add_argument('--max_files', type=int, default=100, help='Max files to load')
    parser.add_argument('--strategy', type=str, default='undersample',
                        choices=['undersample', 'median'],
                        help='Balancing strategy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("="*80)
    print("CLASS BALANCING TEST")
    print("="*80)
    print(f"Algorithm: {args.algorithm}")
    print(f"Strategy: {args.strategy}")
    print(f"Seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    path_glob = f"graph-token/tasks_train/shortest_path/{args.algorithm}/train/*.json"

    examples = load_examples(path_glob, task='shortest_path', data_fraction=0.1, seed=args.seed)

    if not examples:
        print("❌ No examples loaded!")
        return

    # Show original distribution
    original_labels = [ex['label'] for ex in examples if ex.get('label') is not None]
    original_counts = Counter(original_labels)

    print(f"\n{'='*80}")
    print("ORIGINAL DISTRIBUTION")
    print('='*80)
    print(f"Total examples: {len(original_labels)}")

    for label in sorted(original_counts.keys()):
        count = original_counts[label]
        pct = 100 * count / len(original_labels)
        bar = "█" * int(pct / 2)
        print(f"  len{label+1} (class {label}): {count:6d} ({pct:5.1f}%)  {bar}")

    # Apply balancing
    print(f"\n{'='*80}")
    print(f"APPLYING BALANCING (strategy={args.strategy})")
    print('='*80)

    balanced_examples = balance_classes(examples, strategy=args.strategy, seed=args.seed)

    # Show balanced distribution
    balanced_labels = [ex['label'] for ex in balanced_examples if ex.get('label') is not None]
    balanced_counts = Counter(balanced_labels)

    print(f"\n{'='*80}")
    print("BALANCED DISTRIBUTION")
    print('='*80)
    print(f"Total examples: {len(balanced_labels)}")

    for label in sorted(balanced_counts.keys()):
        count = balanced_counts[label]
        pct = 100 * count / len(balanced_labels) if balanced_labels else 0
        bar = "█" * int(pct / 2)
        print(f"  len{label+1} (class {label}): {count:6d} ({pct:5.1f}%)  {bar}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"Original total: {len(original_labels)}")
    print(f"Balanced total: {len(balanced_labels)}")
    print(f"Reduction: {len(original_labels) - len(balanced_labels)} ({100*(1 - len(balanced_labels)/len(original_labels)):.1f}%)")

    # Calculate imbalance ratio
    if balanced_counts:
        max_count = max(balanced_counts.values())
        min_count = min(balanced_counts.values())
        imbalance = max_count / min_count if min_count > 0 else float('inf')
        print(f"Imbalance ratio (max/min): {imbalance:.1f}:1")

        if imbalance <= 1.1:
            print("✓ Classes are now PERFECTLY balanced!")
        elif imbalance <= 2:
            print("✓ Classes are well balanced")
        else:
            print(f"⚠️  Some imbalance remains ({imbalance:.1f}:1)")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

    print("\n💡 To use in training, set in config:")
    print("  dataset:")
    print("    balance_classes: true")
    print(f"    balance_strategy: {args.strategy}")
    print()


if __name__ == "__main__":
    main()
