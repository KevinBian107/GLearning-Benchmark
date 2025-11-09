"""
Analyze actual input formats in the graph-token dataset.

Reports:
1. Token structure and patterns
2. Query format
3. Label/answer format
4. Prediction task types
"""

import os
import sys
import json
from glob import glob
from collections import Counter
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_token_dataset import parse_graph_from_text


def extract_format_pattern(text: str) -> Dict[str, Any]:
    """Extract the actual format pattern from a tokenized sequence."""
    tokens = text.split()

    # Find positions of special tokens
    special_positions = {}
    for tok in ['<bos>', '<n>', '<e>', '<q>', '<p>', '<eos>']:
        if tok in tokens:
            special_positions[tok] = tokens.index(tok)

    # Extract query content
    query_content = []
    if '<q>' in special_positions and '<p>' in special_positions:
        q_idx = special_positions['<q>']
        p_idx = special_positions['<p>']
        query_content = tokens[q_idx + 1:p_idx]

    # Extract answer content
    answer_content = []
    if '<p>' in special_positions:
        p_idx = special_positions['<p>']
        eos_idx = special_positions.get('<eos>', len(tokens))
        answer_content = tokens[p_idx + 1:eos_idx]

    return {
        'special_positions': special_positions,
        'query_content': query_content,
        'answer_content': answer_content,
        'token_count': len(tokens),
    }


def analyze_task(task: str, algorithm: str = 'er', max_samples: int = 50) -> Dict[str, Any]:
    """Analyze actual format in a task's data."""
    print(f"\n{'='*80}")
    print(f"Task: {task} | Algorithm: {algorithm}")
    print(f"{'='*80}\n")

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
        print(f"  ⚠ No data found")
        return {}

    print(f"Analyzing {min(max_samples, len(files))} of {len(files)} files...\n")

    # Collect format patterns
    query_patterns = Counter()
    answer_patterns = Counter()
    token_orders = Counter()

    samples = []
    for fpath in files[:max_samples]:
        with open(fpath) as f:
            data = json.load(f)

        if isinstance(data, list):
            data = data[0] if data else {}

        text = data.get('text', '')
        if not text:
            continue

        pattern = extract_format_pattern(text)
        tokens = text.split()

        # Track token order
        special_order = [tok for tok in tokens if tok in ['<bos>', '<n>', '<e>', '<q>', '<p>', '<eos>']]
        token_orders[' '.join(special_order)] += 1

        # Track query patterns
        query_str = ' '.join(pattern['query_content'][:5])  # First 5 tokens
        query_patterns[query_str] += 1

        # Track answer patterns
        answer_str = ' '.join(pattern['answer_content'][:3])  # First 3 tokens
        answer_patterns[answer_str] += 1

        nodes, edges = parse_graph_from_text(text)
        samples.append({
            'text': text,
            'pattern': pattern,
            'num_nodes': len(nodes) if nodes else 0,
            'num_edges': len(edges),
        })

    # Print findings
    print(f"{'Actual Format':-^80}")
    print(f"\nToken order (most common):")
    for order, count in token_orders.most_common(3):
        print(f"  [{count:3d}x] {order}")

    print(f"\nQuery format (most common):")
    for query, count in query_patterns.most_common(5):
        print(f"  [{count:3d}x] <q> {query}")

    print(f"\nAnswer format (most common):")
    for answer, count in answer_patterns.most_common(5):
        print(f"  [{count:3d}x] <p> {answer}")

    # Show examples
    print(f"\n{'Example Sequences':-^80}")
    for i, sample in enumerate(samples[:2], 1):
        text = sample['text']
        tokens = text.split()
        print(f"\nExample {i} ({len(tokens)} tokens, {sample['num_nodes']} nodes, {sample['num_edges']} edges):")

        # Show truncated sequence with structure
        parts = []
        if '<bos>' in tokens:
            parts.append('<bos>')

        # Find edges section
        edge_end = tokens.index('<n>') if '<n>' in tokens else len(tokens)
        edge_tokens = [t for t in tokens[1:edge_end] if t != '<e>']
        if edge_tokens:
            parts.append(f"{edge_tokens[0]} {edge_tokens[1]} <e> ...")

        # Nodes
        if '<n>' in tokens:
            n_idx = tokens.index('<n>')
            q_idx = tokens.index('<q>') if '<q>' in tokens else len(tokens)
            node_tokens = tokens[n_idx+1:min(n_idx+6, q_idx)]
            parts.append(f"<n> {' '.join(node_tokens)} ...")

        # Query
        if '<q>' in tokens:
            q_idx = tokens.index('<q>')
            p_idx = tokens.index('<p>') if '<p>' in tokens else len(tokens)
            query_tokens = tokens[q_idx:min(q_idx+5, p_idx)]
            parts.append(' '.join(query_tokens))

        # Answer
        if '<p>' in tokens:
            p_idx = tokens.index('<p>')
            answer_tokens = tokens[p_idx:min(p_idx+5, len(tokens))]
            parts.append(' '.join(answer_tokens))

        print(f"  {' '.join(parts)}")

    return {'task': task, 'num_samples': len(samples)}




def main():
    """Analyze actual format patterns in the graph-token dataset."""
    print("\n" + "="*80)
    print("Graph-Token Format Analyzer")
    print("="*80)

    tasks = ['cycle_check', 'edge_existence', 'connected_nodes',
             'node_degree', 'node_count', 'edge_count']
    algorithms = ['er', 'ba', 'sbm']

    results = {}
    for task in tasks:
        for algo in algorithms:
            result = analyze_task(task, algo, max_samples=30)
            if result:
                results[task] = result
                break

    print(f"\n{'='*80}")
    print(f"Summary: {len(results)} tasks analyzed")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
