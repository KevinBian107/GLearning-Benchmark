"""
Publication-quality plotting script for graph learning benchmark results.
Improved color scheme and dataset comparison handling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns
from pathlib import Path
import re

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['grid.alpha'] = 0.3

# Improved color scheme - more distinctive and colorblind-friendly
MODEL_COLORS = {
    'mpnn': '#0173B2',      # Strong Blue
    'gps': '#DE8F05',       # Strong Orange
    'ggps': '#DE8F05',      # Same as GPS
    'ibtt': '#029E73',      # Teal/Green
    'agtt': '#CC78BC',      # Purple/Magenta
}

# Distinctive colors for different training datasets (within same model)
# These are variations of the base color using different hues
DATASET_COLORS = {
    # For MPNN comparisons
    ('mpnn', 'ba+sbm'): '#0173B2',          # Original blue
    ('mpnn', 'er+sbm+path'): '#56B4E9',    # Lighter blue
    ('mpnn', 'path'): '#004D80',            # Darker blue

    # For AGTT comparisons
    ('agtt', 'ba+sbm'): '#CC78BC',          # Original purple
    ('agtt', 'path'): '#E56AAD',            # Pink-purple (bright)
    ('agtt', 'er'): '#7C3F6D',              # Dark purple/wine
    ('agtt', 'er+sbm'): '#9B4F96',          # Medium purple
    ('agtt', 'er+sbm+path'): '#CC78BC',     # Original purple (matches base)

    # For GPS comparisons
    ('gps', 'er+sbm+path'): '#DE8F05',      # Original orange
    ('gps', 'ba+sbm'): '#FDB462',           # Lighter orange

    # For IBTT comparisons
    ('ibtt', 'ba+sbm'): '#029E73',          # Original teal
    ('ibtt', 'path'): '#56C2A3',            # Lighter teal
}

# Line styles
LINE_STYLES = {
    'train': '-',
    'val': '--',
}


def parse_column_name(col_name):
    """
    Parse column name to extract model info and metric type.
    """
    if ' - ' not in col_name:
        return None

    parts = col_name.split(' - ')
    model_part = parts[0]
    metric_part = parts[1] if len(parts) > 1 else ''

    # Extract model type
    model_type = None
    for m in ['mpnn', 'gps', 'ggps', 'ibtt', 'agtt']:
        if f'-{m}-' in model_part.lower():
            model_type = m
            break

    if model_type is None:
        return None

    # Extract dataset from parentheses
    dataset_match = re.search(r'\((.*?)\)', model_part)
    dataset = dataset_match.group(1) if dataset_match else ''

    # Extract task
    task_match = re.search(rf'-{model_type}-(\w+)', model_part.lower())
    task = task_match.group(1) if task_match else ''

    # Extract split and metric
    split = 'train' if 'train/' in metric_part else 'val' if 'val/' in metric_part else ''
    metric = metric_part.split('/')[-1] if '/' in metric_part else ''

    return {
        'model': model_type,
        'task': task,
        'dataset': dataset,
        'split': split,
        'metric': metric,
        'full_name': model_part
    }


def get_color_for_model_dataset(model, dataset):
    """Get appropriate color for model-dataset combination."""
    # Check if specific combination exists
    key = (model.lower(), dataset.lower())
    if key in DATASET_COLORS:
        return DATASET_COLORS[key]

    # Otherwise use base model color
    return MODEL_COLORS.get(model.lower(), '#888888')


def smooth_curve(y, window_length=11, polyorder=3):
    """Smooth curve using Savitzky-Golay filter."""
    if len(y) < window_length:
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        if window_length < polyorder + 2:
            return y

    try:
        return savgol_filter(y, window_length, polyorder)
    except:
        return y


def parse_task_title(filename):
    """
    Parse task title from filename.
    Handle 'extra' files specially to show training dataset comparison.
    """
    stem = filename.stem

    # Check if it's an "extra" comparison file
    if 'extra' in stem:
        # Extract model name
        if 'mpnn' in stem:
            model = 'MPNN'
        elif 'agtt' in stem:
            model = 'AGTT'
        elif 'ibtt' in stem:
            model = 'IBTT'
        elif 'gps' in stem:
            model = 'GPS'
        else:
            model = 'Model'

        # Extract task
        if 'shortest_path' in stem:
            task = 'Shortest Path'
        else:
            task = 'Task'

        # Extract metric
        if 'acc' in stem:
            metric = 'Accuracy'
        elif 'f1' in stem:
            metric = 'F1 Score'
        elif 'loss' in stem:
            metric = 'Loss'
        else:
            metric = 'Metric'

        return f"{task} - {model} Training Dataset Comparison ({metric})"

    # Regular files
    if 'cycle_check' in stem:
        task = 'Cycle Detection'
    elif 'shortest_path' in stem:
        task = 'Shortest Path Prediction'
    elif 'zinc' in stem:
        task = 'ZINC Molecular Property Prediction'
    else:
        task = stem.replace('_', ' ').title()

    if 'acc' in stem:
        metric = 'Accuracy'
    elif 'f1' in stem:
        metric = 'F1 Score'
    elif 'loss' in stem:
        metric = 'Loss'
    else:
        metric = 'Metric'

    return f"{task} - {metric}"


def create_plot(csv_path, output_dir, show_original=True, smooth_window=11):
    """
    Create publication-quality plot from CSV file.
    """
    # Read data
    df = pd.read_csv(csv_path)

    # Get title and metric info
    title = parse_task_title(csv_path)

    # Determine metric type
    if 'loss' in csv_path.stem.lower():
        ylabel = 'Loss'
    elif 'acc' in csv_path.stem.lower():
        ylabel = 'Accuracy'
    elif 'f1' in csv_path.stem.lower():
        ylabel = 'F1 Score'
    else:
        ylabel = 'Metric'

    # Parse columns
    columns_info = {}
    for col in df.columns:
        if col == 'Step' or '__MIN' in col or '__MAX' in col:
            continue

        info = parse_column_name(col)
        if info is None:
            continue

        key = (info['model'], info['dataset'], info['split'])
        columns_info[key] = {
            'col': col,
            'info': info
        }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort to ensure consistent ordering
    for key, col_data in sorted(columns_info.items()):
        model, dataset, split = key
        col_name = col_data['col']
        info = col_data['info']

        # Get data
        x = df['Step'].values
        y = df[col_name].values

        # Remove NaN values
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        # Get color and style
        color = get_color_for_model_dataset(model, dataset)
        linestyle = LINE_STYLES.get(split, '-')
        alpha_original = 0.15 if show_original else 0
        alpha_smooth = 1.0

        # Create label
        model_name = model.upper()
        if dataset:
            # Clean up dataset name
            dataset_clean = dataset.replace('+', ' + ').upper()
            dataset_str = f" ({dataset_clean})"
        else:
            dataset_str = ""
        split_str = split.capitalize()
        label = f"{model_name}{dataset_str} - {split_str}"

        # Plot original data with transparency
        if show_original:
            ax.plot(x, y, color=color, linestyle=linestyle, alpha=alpha_original,
                   linewidth=1.0, zorder=1)

        # Plot smoothed data
        y_smooth = smooth_curve(y, window_length=smooth_window)
        ax.plot(x, y_smooth, color=color, linestyle=linestyle,
               alpha=alpha_smooth, linewidth=2.5, label=label, zorder=2)

        # Error bands
        min_col = col_name + '__MIN'
        max_col = col_name + '__MAX'
        if min_col in df.columns and max_col in df.columns:
            y_min = df[min_col].values[mask]
            y_max = df[max_col].values[mask]

            if not np.allclose(y_min, y_max):
                ax.fill_between(x, y_min, y_max, color=color, alpha=0.12, zorder=0)

    # Formatting
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20, fontsize=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend - adjust columns based on number of entries
    handles, labels = ax.get_legend_handles_labels()
    ncol = 2 if len(handles) > 6 else 1
    ax.legend(handles, labels, loc='best', frameon=True, fancybox=True,
             shadow=True, ncol=ncol, fontsize=9.5)

    # Tight layout
    plt.tight_layout()

    # Save figure (PNG only)
    output_path = output_dir / f"{csv_path.stem}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {csv_path.name} → {output_path.name}")

    plt.close()


def parse_graph_from_text(text):
    """Parse nodes and edges from tokenized graph text."""
    import json
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


def calculate_num_cycles(nodes, edges):
    """Calculate number of cycles using: num_edges - num_nodes + num_components."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    num_components = nx.number_connected_components(G)
    num_cycles = len(edges) - len(nodes) + num_components

    return max(0, num_cycles)


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
    import json
    from collections import defaultdict

    data = defaultdict(list)

    for alg in algorithms:
        task_dir = Path(data_root) / task / alg / 'train'
        if not task_dir.exists():
            print(f"  Warning: {task_dir} not found, skipping {alg}")
            continue

        json_files = sorted(task_dir.glob('*.json'))
        print(f"  Loading {task}/{alg}: {len(json_files)} files")

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


def create_label_distribution_plots(output_dir, data_root='graph-token/tasks_train'):
    """Create label distribution plots for cycle_check and shortest_path tasks."""
    from collections import Counter

    print("\n" + "=" * 80)
    print("Creating Label Distribution Plots")
    print("=" * 80)

    # Check if data directory exists
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        print("Skipping label distribution plots.")
        return

    algorithms = ['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path']

    # Cycle Check - Number of Cycles Distribution
    print("\n[1/2] Creating cycle_check distribution (number of cycles)...")
    cycle_data = load_task_data('cycle_check', algorithms, data_root)

    if cycle_data:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Collect all cycle counts by algorithm
        all_data = []
        labels = []
        colors_list = []

        for alg in algorithms:
            if alg not in cycle_data or len(cycle_data[alg]) == 0:
                continue

            cycles = [calculate_num_cycles(d['nodes'], d['edges']) for d in cycle_data[alg]]
            all_data.append(cycles)
            labels.append(f"{alg.upper()}\n(n={len(cycles)})")
            colors_list.append(plt.cm.Set3(len(all_data) / len(algorithms)))

        if all_data:
            # Create violin plot
            positions = list(range(len(all_data)))
            parts = ax.violinplot(all_data, positions=positions, showmeans=True,
                                   showmedians=True, widths=0.7)

            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Number of Cycles', fontweight='bold', fontsize=14)
            ax.set_title('Cycle Check - Distribution of Number of Cycles by Algorithm',
                        fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            output_path = output_dir / 'cycle_check_num_cycles_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path.name}")
            plt.close()

    # Shortest Path - Length Distribution
    print("\n[2/2] Creating shortest_path distribution (path lengths)...")
    sp_data = load_task_data('shortest_path', algorithms, data_root)

    if sp_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Line plot by algorithm
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))

        for i, alg in enumerate(algorithms):
            if alg not in sp_data or len(sp_data[alg]) == 0:
                continue

            lengths = [d['label'] for d in sp_data[alg] if d['label'] is not None]
            if lengths:
                counts = Counter(lengths)
                sorted_lengths = sorted(counts.keys())
                sorted_counts = [counts[l] for l in sorted_lengths]

                ax1.plot(sorted_lengths, sorted_counts, marker='o', label=alg.upper(),
                        linewidth=2.5, markersize=8, alpha=0.8, color=colors[i])

        ax1.set_xlabel('Shortest Path Length', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Count', fontweight='bold', fontsize=14)
        ax1.set_title('Path Length Distribution by Algorithm',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', ncol=2, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Plot 2: Combined bar chart
        all_lengths = []
        for alg in algorithms:
            if alg not in sp_data:
                continue
            lengths = [d['label'] for d in sp_data[alg] if d['label'] is not None]
            all_lengths.extend(lengths)

        if all_lengths:
            length_counts = Counter(all_lengths)
            sorted_lengths = sorted(length_counts.keys())
            counts = [length_counts[l] for l in sorted_lengths]

            bars = ax2.bar(sorted_lengths, counts, color='steelblue', alpha=0.7,
                          edgecolor='black', linewidth=1.5)

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom',
                        fontweight='bold', fontsize=10)

            ax2.set_xlabel('Shortest Path Length (Class)', fontweight='bold', fontsize=14)
            ax2.set_ylabel('Total Count', fontweight='bold', fontsize=14)
            ax2.set_title('Combined Path Length Distribution\n(All Algorithms)',
                         fontsize=14, fontweight='bold')
            ax2.set_xticks(sorted_lengths)
            ax2.set_xticklabels([f'len{l}' for l in sorted_lengths])
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            # Add stats
            total = sum(counts)
            mean_len = np.mean(all_lengths)
            median_len = np.median(all_lengths)
            stats_text = f'Total: {total} | Mean: {mean_len:.2f} | Median: {median_len:.1f}'
            ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes,
                    ha='center', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        output_path = output_dir / 'shortest_path_length_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()

    print("\n" + "=" * 80)


def create_zinc_distribution_plots(output_dir, zinc_root='./data/ZINC'):
    """Create ZINC molecular dataset distribution plots."""
    print("\n" + "=" * 80)
    print("Creating ZINC Dataset Distribution Plots")
    print("=" * 80)

    # Check if ZINC data exists
    from pathlib import Path as PathLib
    if not PathLib(zinc_root).exists():
        print(f"ZINC data directory not found: {zinc_root}")
        print("Skipping ZINC distribution plots.")
        print("To generate ZINC plots, first download the dataset by running a ZINC training script.")
        return

    try:
        from torch_geometric.datasets import ZINC
        import torch
        from collections import Counter
    except ImportError:
        print("PyTorch Geometric not installed. Skipping ZINC distribution plots.")
        return

    print("\nLoading ZINC 12K subset...")
    try:
        train_dataset = ZINC(root=zinc_root, subset=True, split='train')
        val_dataset = ZINC(root=zinc_root, subset=True, split='val')
        test_dataset = ZINC(root=zinc_root, subset=True, split='test')
    except Exception as e:
        print(f"Error loading ZINC dataset: {e}")
        print("Skipping ZINC distribution plots.")
        return

    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    print(f"Dataset sizes:")
    for split, dataset in datasets.items():
        print(f"  {split.upper()}: {len(dataset)} molecular graphs")

    # Extract statistics
    print("\nComputing statistics...")
    stats = {}
    for split, dataset in datasets.items():
        num_nodes_list = []
        num_edges_list = []
        targets = []

        for data in dataset:
            num_nodes_list.append(data.num_nodes)
            num_edges_list.append(data.num_edges)
            targets.append(data.y.item())

        stats[split] = {
            'num_nodes': num_nodes_list,
            'num_edges': num_edges_list,
            'targets': targets
        }

        print(f"  {split.upper()}: avg atoms={np.mean(num_nodes_list):.1f}, "
              f"avg bonds={np.mean(num_edges_list):.1f}, "
              f"solubility range=[{min(targets):.3f}, {max(targets):.3f}]")

    # Create simple 1x2 distribution plot
    print("\nCreating distribution plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    splits = ['train', 'val']
    colors = ['#0173B2', '#DE8F05']  # Blue, Orange

    # Plot 1: Train Solubility Distribution
    ax = ax1
    targets = stats['train']['targets']
    ax.hist(targets, bins=50, alpha=0.7, color=colors[0], edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Constrained Solubility', fontweight='bold', fontsize=14)
    ax.set_ylabel('Count', fontweight='bold', fontsize=14)
    ax.set_title('Train Set Solubility Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add statistics
    target_min, target_max = min(targets), max(targets)
    target_mean, target_std = np.mean(targets), np.std(targets)
    stats_text = f'N = {len(targets)}\nRange: [{target_min:.2f}, {target_max:.2f}]\nMean: {target_mean:.2f}\nStd: {target_std:.2f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           ha='right', va='top', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

    # Plot 2: Val Solubility Distribution
    ax = ax2
    targets = stats['val']['targets']
    ax.hist(targets, bins=50, alpha=0.7, color=colors[1], edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Constrained Solubility', fontweight='bold', fontsize=14)
    ax.set_ylabel('Count', fontweight='bold', fontsize=14)
    ax.set_title('Validation Set Solubility Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add statistics
    target_min, target_max = min(targets), max(targets)
    target_mean, target_std = np.mean(targets), np.std(targets)
    stats_text = f'N = {len(targets)}\nRange: [{target_min:.2f}, {target_max:.2f}]\nMean: {target_mean:.2f}\nStd: {target_std:.2f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           ha='right', va='top', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

    # Overall title
    fig.suptitle('ZINC 12K Dataset - Solubility Distribution',
                fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()
    output_path = output_dir / 'zinc_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()

    print("\n" + "=" * 80)


def main():
    """Process all CSV files in the data directory."""
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'figures_data'
    output_dir = script_dir / 'figures_output'

    # Path to graph-token data (relative to script location)
    graph_token_data = script_dir.parent / 'graph-token' / 'tasks_train'

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get CSV files
    csv_files = list(data_dir.glob('*.csv'))

    if not csv_files:
        print("No CSV files found in data/ directory")
        return

    print("=" * 80)
    print(f"Graph Learning Benchmark - Publication Plots")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(csv_files)} CSV files\n")

    for csv_file in sorted(csv_files):
        try:
            create_plot(csv_file, output_dir, show_original=True, smooth_window=11)
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"✓ All training curve plots saved to: {output_dir}")
    print("=" * 80)

    # Create label distribution plots
    create_label_distribution_plots(output_dir, data_root=str(graph_token_data))

    # Create ZINC distribution plots
    zinc_data_path = script_dir.parent / 'data' / 'ZINC'
    create_zinc_distribution_plots(output_dir, zinc_root=str(zinc_data_path))


if __name__ == "__main__":
    main()
