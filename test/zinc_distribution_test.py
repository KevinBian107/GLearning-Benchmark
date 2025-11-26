"""
ZINC molecular graph dataset distribution analysis and visualization.

ZINC 12K subset: Graph-level regression task for molecular property prediction.
Task: Predict constrained solubility (continuous real-valued target).
Dataset: 12,000 molecular graphs from "Benchmarking Graph Neural Networks" paper.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from torch_geometric.datasets import ZINC
    import torch
except ImportError:
    print("Error: PyTorch Geometric not installed. Run: pip install torch-geometric")
    sys.exit(1)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10


def load_zinc_data(root='./data/ZINC', subset=True):
    """Load ZINC 12K subset for graph regression."""
    print(f"Loading ZINC dataset (subset={subset})...")
    if subset:
        print("  Using ZINC 12K subset from 'Benchmarking Graph Neural Networks' paper")

    train_dataset = ZINC(root=root, subset=subset, split='train')
    val_dataset = ZINC(root=root, subset=subset, split='val')
    test_dataset = ZINC(root=root, subset=subset, split='test')

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


def compute_statistics(dataset):
    """Extract graph statistics from dataset."""
    num_nodes_list = []
    num_edges_list = []
    degree_list = []
    targets = []

    for data in dataset:
        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.num_edges)
        targets.append(data.y.item())

        edge_index = data.edge_index
        degrees = torch.bincount(edge_index[0], minlength=data.num_nodes)
        degree_list.extend(degrees.tolist())

    return {
        'num_nodes': num_nodes_list,
        'num_edges': num_edges_list,
        'degrees': degree_list,
        'targets': targets
    }


def create_zinc_analysis_plot(datasets):
    """Create comprehensive analysis visualization."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    splits = ['train', 'val', 'test']
    stats = {split: compute_statistics(datasets[split]) for split in splits}

    print("Computing statistics...")
    for split in splits:
        s = stats[split]
        print(f"{split.upper()}: {len(datasets[split])} graphs, "
              f"avg nodes={np.mean(s['num_nodes']):.1f}, "
              f"avg edges={np.mean(s['num_edges']):.1f}, "
              f"target range=[{min(s['targets']):.2f}, {max(s['targets']):.2f}]")

    # Row 0: Number of nodes distribution
    print("Plotting node distribution...")
    ax = fig.add_subplot(gs[0, :])

    colors = ['steelblue', 'coral', 'seagreen']
    for i, split in enumerate(splits):
        counts = Counter(stats[split]['num_nodes'])
        sorted_nodes = sorted(counts.keys())
        sorted_counts = [counts[n] for n in sorted_nodes]

        ax.plot(sorted_nodes, sorted_counts, marker='o', label=split.upper(),
               linewidth=2, markersize=6, alpha=0.7, color=colors[i])

    ax.set_xlabel('Number of Atoms (Nodes)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax.set_title('Molecular Graph Size Distribution - Number of Atoms',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Row 1: Number of edges distribution
    print("Plotting edge distribution...")
    ax = fig.add_subplot(gs[1, :])

    for i, split in enumerate(splits):
        counts = Counter(stats[split]['num_edges'])
        sorted_edges = sorted(counts.keys())
        sorted_counts = [counts[e] for e in sorted_edges]

        ax.plot(sorted_edges, sorted_counts, marker='o', label=split.upper(),
               linewidth=2, markersize=6, alpha=0.7, color=colors[i])

    ax.set_xlabel('Number of Chemical Bonds (Edges)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax.set_title('Molecular Graph Size Distribution - Number of Bonds',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Row 2: Degree distribution
    print("Plotting degree distribution...")
    ax = fig.add_subplot(gs[2, :])

    for i, split in enumerate(splits):
        degree_counts = Counter(stats[split]['degrees'])
        max_degree = min(max(degree_counts.keys()), 10)
        degrees = range(max_degree + 1)
        counts = [degree_counts.get(d, 0) for d in degrees]

        ax.bar([d + i*0.25 for d in degrees], counts, width=0.25,
               label=split.upper(), alpha=0.7, color=colors[i])

    ax.set_xlabel('Node Degree', fontweight='bold', fontsize=12)
    ax.set_ylabel('Count (log scale)', fontweight='bold', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Node Degree Distribution by Split',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3, Col 0: Target regression value distribution
    print("Plotting target value distribution...")
    ax = fig.add_subplot(gs[3, 0])

    all_targets = []
    for i, split in enumerate(splits):
        targets = stats[split]['targets']
        all_targets.extend(targets)
        ax.hist(targets, bins=50, alpha=0.6,
               label=split.upper(), color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Constrained Solubility (Regression Target)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax.set_title('Regression Target Distribution (Constrained Solubility)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    target_min, target_max = min(all_targets), max(all_targets)
    target_mean, target_std = np.mean(all_targets), np.std(all_targets)
    ax.text(0.98, 0.97, f'Range: [{target_min:.2f}, {target_max:.2f}]\nMean: {target_mean:.2f}, Std: {target_std:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Row 3, Col 1: Box plots for nodes/edges
    print("Creating box plots...")
    ax = fig.add_subplot(gs[3, 1])

    node_data = [stats[split]['num_nodes'] for split in splits]
    positions = np.arange(len(splits))

    bp = ax.boxplot(node_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([s.upper() for s in splits])
    ax.set_ylabel('Number of Atoms', fontweight='bold', fontsize=11)
    ax.set_title('Molecular Size Distribution (Atoms)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3, Col 2: Statistics table
    print("Creating statistics table...")
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')

    stats_text = []
    stats_text.append("=" * 60)
    stats_text.append(f"{'Split':<8} {'Samples':<10} {'Avg Atoms':<12} {'Avg Bonds':<12} {'Avg Target':<12}")
    stats_text.append("=" * 60)

    for split in splits:
        s = stats[split]
        avg_nodes = np.mean(s['num_nodes'])
        avg_edges = np.mean(s['num_edges'])
        avg_target = np.mean(s['targets'])

        stats_text.append(
            f"{split.upper():<8} {len(datasets[split]):<10} "
            f"{avg_nodes:<12.2f} {avg_edges:<12.2f} {avg_target:<12.4f}"
        )

    stats_text.append("=" * 60)
    stats_text.append("")
    stats_text.append("Regression Target: Constrained Solubility")
    stats_text.append("-" * 60)

    all_targets = []
    for split in splits:
        targets = stats[split]['targets']
        all_targets.extend(targets)
        stats_text.append(
            f"{split.upper()}: min={min(targets):.4f}, "
            f"max={max(targets):.4f}, "
            f"mean={np.mean(targets):.4f}, "
            f"std={np.std(targets):.4f}"
        )

    stats_text.append("-" * 60)
    stats_text.append(f"Overall: min={min(all_targets):.4f}, max={max(all_targets):.4f}")
    stats_text.append("=" * 60)
    stats_text.append("Task: Graph-level regression")
    stats_text.append("Metric: Mean Absolute Error (MAE)")

    ax.text(0.5, 0.5, '\n'.join(stats_text),
           ha='center', va='center', fontsize=9.5, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('ZINC 12K - Molecular Graph Regression Dataset - Distribution Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def plot_sample_molecules(datasets, num_samples=6):
    """Visualize sample molecular graphs."""
    try:
        import networkx as nx
    except ImportError:
        print("Warning: NetworkX not installed, skipping molecule visualization")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    train_dataset = datasets['train']

    print(f"Creating {num_samples} sample molecule visualizations...")

    indices = np.linspace(0, len(train_dataset)-1, num_samples, dtype=int)

    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off')
            continue

        data = train_dataset[indices[idx]]

        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))

        edge_index = data.edge_index.numpy()
        edges = [(edge_index[0, i], edge_index[1, i])
                 for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, seed=42, k=0.5)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=200, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

        ax.set_title(f'Molecule {indices[idx]} (N={data.num_nodes}, E={data.num_edges}, y={data.y.item():.3f})',
                    fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Sample Molecular Graphs from ZINC 12K Dataset',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def main():
    """Main analysis function."""
    print("=" * 80)
    print("ZINC 12K Dataset Analysis - Graph Regression Task")
    print("=" * 80)
    print("Task: Predict constrained solubility (regression)")
    print("Paper: Benchmarking Graph Neural Networks (2020)")
    print("=" * 80)

    output_dir = Path('test/output')
    output_dir.mkdir(exist_ok=True)

    print("\nLoading ZINC 12K subset...")
    datasets = load_zinc_data(subset=True)

    print(f"\nDataset sizes:")
    for split, dataset in datasets.items():
        print(f"  {split.upper()}: {len(dataset)} molecular graphs")

    print("\n[1/2] Creating distribution analysis plot...")
    fig1 = create_zinc_analysis_plot(datasets)
    output_path1 = output_dir / 'zinc_distribution.png'
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path1}")
    plt.close(fig1)

    print("\n[2/2] Creating sample molecule visualizations...")
    fig2 = plot_sample_molecules(datasets, num_samples=6)
    if fig2 is not None:
        output_path2 = output_dir / 'zinc_sample_molecules.png'
        fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path2}")
        plt.close(fig2)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
