"""
Data distribution analysis for cycle_check and shortest_path tasks.
Analyzes training data across all graph generation algorithms.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


def parse_graph_from_text(text):
    """Parse nodes and edges from tokenized graph text."""
    tokens = text.split()
    edges = []
    nodes = []

    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i+2] == '<e>':
            try:
                src = int(tokens[i])
                tgt = int(tokens[i + 1])
                edges.append((src, tgt))
                i += 3
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

    return nodes, edges


def parse_cycle_check_label(text):
    """Parse yes/no label from cycle_check text."""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok == '<p>' and i + 1 < len(tokens):
            label_tok = tokens[i + 1].lower()
            if label_tok == 'yes':
                return 1
            elif label_tok == 'no':
                return 0
    return None


def parse_shortest_path_label(text):
    """Parse distance label from shortest_path text."""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok == '<p>' and i + 1 < len(tokens):
            label_tok = tokens[i + 1].upper()
            if label_tok in ('INF', 'INFINITY', '<EOS>'):
                return None  # Unreachable
            if label_tok.startswith('LEN'):
                try:
                    distance = int(label_tok[3:])
                    return distance
                except ValueError:
                    pass
    return None


def load_task_data(task, algorithms, data_root='graph-token/tasks_train'):
    """Load all data for a given task across algorithms."""
    data = defaultdict(list)

    for alg in algorithms:
        task_dir = Path(data_root) / task / alg / 'train'
        if not task_dir.exists():
            print(f"Warning: {task_dir} not found, skipping {alg}")
            continue

        json_files = sorted(task_dir.glob('*.json'))
        print(f"Loading {task}/{alg}: {len(json_files)} files")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    content = json.load(f)
                except json.JSONDecodeError:
                    continue

            # Handle both list and single dict formats
            records = content if isinstance(content, list) else [content]

            for record in records:
                text = record.get('text', '')
                if not text:
                    continue

                nodes, edges = parse_graph_from_text(text)
                if not nodes:
                    continue

                # Parse label based on task
                if task == 'cycle_check':
                    label = parse_cycle_check_label(text)
                elif task == 'shortest_path':
                    label = parse_shortest_path_label(text)
                else:
                    label = None

                data[alg].append({
                    'nodes': nodes,
                    'edges': edges,
                    'label': label,
                    'text': text
                })

    return data


def calculate_num_cycles(nodes, edges):
    """Calculate number of cycles using: num_edges - num_nodes + num_components."""
    # Build graph to count components
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    num_components = nx.number_connected_components(G)
    num_cycles = len(edges) - len(nodes) + num_components

    return max(0, num_cycles)  # Can't be negative


def create_example_graph(nodes, edges, title):
    """Create a sample graph visualization."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Use spring layout for nice visualization
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=300, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title(title, fontsize=11, fontweight='bold')
    plt.axis('off')


def analyze_cycle_check(data, algorithms):
    """Create comprehensive analysis plot for cycle_check task."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, len(algorithms), hspace=0.4, wspace=0.3)

    # Row 0: Example graphs
    print("Creating example graphs...")
    for col, alg in enumerate(algorithms):
        if alg not in data or len(data[alg]) == 0:
            continue

        ax = fig.add_subplot(gs[0, col])
        plt.sca(ax)

        # Get first example
        example = data[alg][0]
        create_example_graph(example['nodes'], example['edges'],
                           f"{alg.upper()} (N={len(example['nodes'])}, E={len(example['edges'])})")

    # Row 1: Number of cycles distribution
    print("Plotting cycle distribution...")
    ax = fig.add_subplot(gs[1, :])

    all_cycles = []
    alg_labels = []
    for alg in algorithms:
        if alg not in data:
            continue
        cycles = [calculate_num_cycles(d['nodes'], d['edges']) for d in data[alg]]
        all_cycles.extend(cycles)
        alg_labels.extend([alg] * len(cycles))

    # Violin plot
    positions = []
    cycle_data = []
    labels = []
    for i, alg in enumerate(algorithms):
        if alg not in data:
            continue
        cycles = [calculate_num_cycles(d['nodes'], d['edges']) for d in data[alg]]
        cycle_data.append(cycles)
        positions.append(i)
        labels.append(f"{alg}\n(n={len(cycles)})")

    parts = ax.violinplot(cycle_data, positions=positions, showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of Cycles', fontweight='bold')
    ax.set_title('Distribution of Number of Cycles by Graph Algorithm',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Row 2: Node and edge distributions
    ax1 = fig.add_subplot(gs[2, :len(algorithms)//2 + len(algorithms)%2])
    ax2 = fig.add_subplot(gs[2, len(algorithms)//2 + len(algorithms)%2:])

    # Nodes
    node_data = []
    for i, alg in enumerate(algorithms):
        if alg not in data:
            continue
        nodes_counts = [len(d['nodes']) for d in data[alg]]
        node_data.append(nodes_counts)

    parts = ax1.violinplot(node_data, positions=positions, showmeans=True, showmedians=True)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Number of Nodes', fontweight='bold')
    ax1.set_title('Distribution of Graph Sizes (Nodes)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Edges
    edge_data = []
    for i, alg in enumerate(algorithms):
        if alg not in data:
            continue
        edge_counts = [len(d['edges']) for d in data[alg]]
        edge_data.append(edge_counts)

    parts = ax2.violinplot(edge_data, positions=positions, showmeans=True, showmedians=True)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Number of Edges', fontweight='bold')
    ax2.set_title('Distribution of Graph Sizes (Edges)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Row 3: Label distribution (has_cycle yes/no)
    ax = fig.add_subplot(gs[3, :])

    label_counts = defaultdict(lambda: {'yes': 0, 'no': 0})
    for alg in algorithms:
        if alg not in data:
            continue
        for d in data[alg]:
            if d['label'] == 1:
                label_counts[alg]['yes'] += 1
            elif d['label'] == 0:
                label_counts[alg]['no'] += 1

    x = np.arange(len(algorithms))
    width = 0.35

    yes_counts = [label_counts[alg]['yes'] for alg in algorithms]
    no_counts = [label_counts[alg]['no'] for alg in algorithms]

    ax.bar(x - width/2, yes_counts, width, label='Has Cycle (Yes)', color='coral', alpha=0.8)
    ax.bar(x + width/2, no_counts, width, label='No Cycle (No)', color='skyblue', alpha=0.8)

    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Cycle Check Label Distribution by Algorithm', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([alg.upper() for alg in algorithms])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cycle Check Task - Data Distribution Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def analyze_shortest_path(data, algorithms, combined_dist_algorithms=None):
    """Create comprehensive analysis plot for shortest_path task.

    Args:
        data: Dictionary of data by algorithm
        algorithms: List of all algorithms to display
        combined_dist_algorithms: List of algorithms to include in combined distribution.
                                  If None, uses all algorithms.
    """
    if combined_dist_algorithms is None:
        combined_dist_algorithms = algorithms

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(5, len(algorithms), hspace=0.4, wspace=0.3)

    # Row 0: Example graphs
    print("Creating example graphs...")
    for col, alg in enumerate(algorithms):
        if alg not in data or len(data[alg]) == 0:
            continue

        ax = fig.add_subplot(gs[0, col])
        plt.sca(ax)

        # Get first example
        example = data[alg][0]
        create_example_graph(example['nodes'], example['edges'],
                           f"{alg.upper()} (N={len(example['nodes'])}, E={len(example['edges'])})")

    # Row 1: Shortest path length distribution
    print("Plotting path length distribution...")
    ax = fig.add_subplot(gs[1, :])

    # Histogram for all algorithms
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))

    for i, alg in enumerate(algorithms):
        if alg not in data:
            continue

        lengths = [d['label'] for d in data[alg] if d['label'] is not None]
        if lengths:
            counts = Counter(lengths)
            sorted_lengths = sorted(counts.keys())
            sorted_counts = [counts[l] for l in sorted_lengths]

            ax.plot(sorted_lengths, sorted_counts, marker='o', label=alg.upper(),
                   linewidth=2, markersize=8, alpha=0.7, color=colors[i])

    ax.set_xlabel('Shortest Path Length', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Distribution of Shortest Path Lengths by Algorithm',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', ncol=len(algorithms))
    ax.grid(True, alpha=0.3)

    # Row 2: Combined class distribution across specified algorithms
    print("Plotting combined class distribution...")
    ax = fig.add_subplot(gs[2, :])

    # Collect all path lengths from specified algorithms only
    all_lengths = []
    included_algs = []
    for alg in combined_dist_algorithms:
        if alg not in data:
            continue
        lengths = [d['label'] for d in data[alg] if d['label'] is not None]
        all_lengths.extend(lengths)
        included_algs.append(alg.upper())

    if all_lengths:
        # Count occurrences of each path length
        length_counts = Counter(all_lengths)
        sorted_lengths = sorted(length_counts.keys())
        counts = [length_counts[l] for l in sorted_lengths]

        # Create bar chart
        bars = ax.bar(sorted_lengths, counts, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_xlabel('Shortest Path Length (Class)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Total Count', fontweight='bold', fontsize=12)

        # Build title with list of included algorithms
        alg_list = ', '.join(included_algs)
        title = f'Combined Class Distribution\nAlgorithms: [{alg_list}]'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_xticks(sorted_lengths)
        ax.set_xticklabels([f'len{l}' for l in sorted_lengths])
        ax.grid(True, alpha=0.3, axis='y')

        # Add summary statistics
        total = sum(counts)
        min_len = min(sorted_lengths)
        max_len = max(sorted_lengths)
        mean_len = np.mean(all_lengths)
        median_len = np.median(all_lengths)

        stats_text = f'Total samples: {total} | Min: len{min_len} | Max: len{max_len} | Mean: {mean_len:.2f} | Median: {median_len:.1f}'
        ax.text(0.5, 0.95, stats_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Row 3: Node and edge distributions
    ax1 = fig.add_subplot(gs[3, :len(algorithms)//2 + len(algorithms)%2])
    ax2 = fig.add_subplot(gs[3, len(algorithms)//2 + len(algorithms)%2:])

    positions = []
    labels = []
    for i, alg in enumerate(algorithms):
        if alg not in data:
            continue
        positions.append(i)
        labels.append(f"{alg}\n(n={len(data[alg])})")

    # Nodes
    node_data = []
    for alg in algorithms:
        if alg not in data:
            continue
        # Get unique graphs (not all pairs)
        unique_graphs = {}
        for d in data[alg]:
            graph_key = (tuple(sorted(d['nodes'])), tuple(sorted(d['edges'])))
            if graph_key not in unique_graphs:
                unique_graphs[graph_key] = len(d['nodes'])
        node_data.append(list(unique_graphs.values()))

    parts = ax1.violinplot(node_data, positions=positions, showmeans=True, showmedians=True)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Number of Nodes', fontweight='bold')
    ax1.set_title('Distribution of Graph Sizes (Nodes)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Edges
    edge_data = []
    for alg in algorithms:
        if alg not in data:
            continue
        unique_graphs = {}
        for d in data[alg]:
            graph_key = (tuple(sorted(d['nodes'])), tuple(sorted(d['edges'])))
            if graph_key not in unique_graphs:
                unique_graphs[graph_key] = len(d['edges'])
        edge_data.append(list(unique_graphs.values()))

    parts = ax2.violinplot(edge_data, positions=positions, showmeans=True, showmedians=True)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Number of Edges', fontweight='bold')
    ax2.set_title('Distribution of Graph Sizes (Edges)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Row 4: Statistics summary
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')

    # Compute statistics
    stats_text = []
    stats_text.append("=" * 100)
    stats_text.append(f"{'Algorithm':<15} {'Samples':<10} {'Avg Path Len':<15} {'Min/Max Len':<15} {'Avg Nodes':<12} {'Avg Edges':<12}")
    stats_text.append("=" * 100)

    for alg in algorithms:
        if alg not in data:
            continue

        lengths = [d['label'] for d in data[alg] if d['label'] is not None]
        unique_graphs = {}
        for d in data[alg]:
            graph_key = (tuple(sorted(d['nodes'])), tuple(sorted(d['edges'])))
            if graph_key not in unique_graphs:
                unique_graphs[graph_key] = (len(d['nodes']), len(d['edges']))

        nodes_list = [g[0] for g in unique_graphs.values()]
        edges_list = [g[1] for g in unique_graphs.values()]

        avg_len = np.mean(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        avg_nodes = np.mean(nodes_list) if nodes_list else 0
        avg_edges = np.mean(edges_list) if edges_list else 0

        stats_text.append(
            f"{alg.upper():<15} {len(data[alg]):<10} {avg_len:<15.2f} "
            f"{min_len}/{max_len:<13} {avg_nodes:<12.2f} {avg_edges:<12.2f}"
        )

    stats_text.append("=" * 100)

    ax.text(0.5, 0.5, '\n'.join(stats_text),
           ha='center', va='center', fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Shortest Path Task - Data Distribution Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    """Main analysis function."""
    # Algorithms to analyze (all available)
    algorithms = ['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path']

    # Algorithms to include in combined class distribution
    # (typically matches your training configuration)
    combined_dist_algorithms = ['er', 'ba', 'sbm', 'path']  # Training algorithms

    # Output directory
    output_dir = Path('test/output')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Data Distribution Analysis")
    print("=" * 80)
    print(f"\nAll algorithms: {algorithms}")
    print(f"Combined distribution algorithms: {combined_dist_algorithms}")

    # Analyze cycle_check
    print("\n[1/2] Analyzing cycle_check task...")
    cycle_data = load_task_data('cycle_check', algorithms)

    if cycle_data:
        fig = analyze_cycle_check(cycle_data, algorithms)
        output_path = output_dir / 'cycle_check_distribution.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)
    else:
        print("No data found for cycle_check")

    # Analyze shortest_path
    print("\n[2/2] Analyzing shortest_path task...")
    sp_data = load_task_data('shortest_path', algorithms)

    if sp_data:
        fig = analyze_shortest_path(sp_data, algorithms, combined_dist_algorithms)
        output_path = output_dir / 'shortest_path_distribution.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)
    else:
        print("No data found for shortest_path")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
