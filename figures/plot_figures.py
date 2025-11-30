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
    ('agtt', 'path'): '#E56AAD',            # Pink-purple
    ('agtt', 'er+sbm'): '#9B4F96',          # Darker purple

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


def main():
    """Process all CSV files in the data directory."""
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'figures_data'
    output_dir = script_dir / 'figures_output'

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
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
