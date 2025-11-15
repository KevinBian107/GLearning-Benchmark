"""
Shared metrics computation for all training scripts.

Provides comprehensive metrics for both cycle_check (binary) and shortest_path (multi-class) tasks.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from PIL import Image


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task: str = 'cycle_check',
    loss_val: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a batch of predictions.

    Args:
        logits: Model outputs (batch_size, num_classes) or (batch_size,) for binary
        labels: Ground truth labels (batch_size,)
        task: 'cycle_check' or 'shortest_path'
        loss_val: Optional loss value to include in metrics

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, confusion_matrix, etc.
    """
    metrics = {}

    # Move to CPU and convert to numpy
    if logits.dim() > 1:
        preds = logits.argmax(dim=-1).cpu().numpy()
    else:
        # Binary classification with single logit
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()

    labels_np = labels.cpu().numpy()

    # Basic accuracy
    accuracy = (preds == labels_np).mean()
    metrics['accuracy'] = float(accuracy)

    # For shortest_path: MSE and MAE (treating class indices as ordinal values)
    if task == 'shortest_path':
        mse = ((preds - labels_np) ** 2).mean()
        mae = np.abs(preds - labels_np).mean()
        metrics['mse'] = float(mse)
        metrics['mae'] = float(mae)

    # Confusion matrix
    if task == 'cycle_check':
        num_classes = 2
    else:  # shortest_path
        num_classes = 7  # len1-len7

    cm = confusion_matrix(labels_np, preds, labels=list(range(num_classes)))
    metrics['confusion_matrix'] = cm  # Keep as numpy array for logging

    # Precision, Recall, F1
    # For binary: average='binary'
    # For multi-class: average='macro' (unweighted mean) and 'weighted' (weighted by support)
    if task == 'cycle_check':
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average='binary', zero_division=0
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
    else:
        # Macro-averaged (equal weight to each class)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels_np, preds, average='macro', zero_division=0
        )
        metrics['precision_macro'] = float(precision_macro)
        metrics['recall_macro'] = float(recall_macro)
        metrics['f1_macro'] = float(f1_macro)

        # Weighted-averaged (weighted by support)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels_np, preds, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = float(precision_weighted)
        metrics['recall_weighted'] = float(recall_weighted)
        metrics['f1_weighted'] = float(f1_weighted)

    # Add loss if provided
    if loss_val is not None:
        metrics['loss'] = float(loss_val)

    return metrics


def aggregate_metrics(metrics_list: list) -> Dict[str, float]:
    """
    Aggregate metrics from multiple batches (for epoch-level metrics).

    Args:
        metrics_list: List of metric dictionaries from compute_metrics()

    Returns:
        Aggregated metrics dictionary with mean values
    """
    if not metrics_list:
        return {}

    # Collect all metric keys (excluding confusion_matrix which needs special handling)
    metric_keys = set()
    for m in metrics_list:
        metric_keys.update(k for k in m.keys() if k != 'confusion_matrix')

    aggregated = {}

    # Average numerical metrics
    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        aggregated[key] = float(np.mean(values))

    # Sum confusion matrices
    if 'confusion_matrix' in metrics_list[0]:
        cm_sum = sum(m['confusion_matrix'] for m in metrics_list)
        aggregated['confusion_matrix'] = cm_sum

    return aggregated


def format_confusion_matrix(cm: np.ndarray, task: str = 'cycle_check') -> str:
    """
    Format confusion matrix as a readable string.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        task: 'cycle_check' or 'shortest_path'

    Returns:
        Formatted string representation
    """
    if task == 'cycle_check':
        labels = ['No', 'Yes']
    else:
        labels = [f'len{i+1}' for i in range(7)]

    # Header
    header = "Confusion Matrix:\n"
    header += "Predicted →\n"
    header += "Actual ↓  " + "  ".join(f"{l:>6}" for l in labels) + "\n"

    # Rows
    rows = []
    for i, label in enumerate(labels):
        row = f"{label:>6}  " + "  ".join(f"{cm[i, j]:>6}" for j in range(len(labels)))
        rows.append(row)

    return header + "\n".join(rows)


def get_loss_function(task: str, device: torch.device):
    """
    Get appropriate loss function for the task.

    Args:
        task: 'cycle_check' or 'shortest_path'
        device: torch device

    Returns:
        Loss function (callable)
    """
    if task == 'cycle_check':
        # Binary classification: CrossEntropyLoss with 2 classes
        return torch.nn.CrossEntropyLoss()
    else:
        # shortest_path: Use CrossEntropyLoss for training (stronger gradients)
        # MSE/MAE are still computed as metrics for evaluation
        # Treating shortest path as multi-class classification (len1-len7)
        return torch.nn.CrossEntropyLoss()


def log_graph_examples(dataset, task: str, num_examples: int = 2) -> str:
    """
    Create a string representation of example graphs for logging.

    Args:
        dataset: PyG dataset (for MPNN/GPS)
        task: 'cycle_check' or 'shortest_path'
        num_examples: Number of examples to show

    Returns:
        Formatted string with graph examples
    """
    examples_str = f"\n{'='*80}\n"
    examples_str += f"Example Graphs ({task})\n"
    examples_str += f"{'='*80}\n\n"

    for i in range(min(num_examples, len(dataset))):
        data = dataset[i]
        examples_str += f"Example {i+1}:\n"
        examples_str += f"  Nodes: {data.num_nodes}\n"
        examples_str += f"  Edges: {data.edge_index.size(1)}\n"

        if task == 'cycle_check':
            label_str = "Yes (has cycle)" if data.y.item() == 1 else "No (no cycle)"
            examples_str += f"  Label: {label_str}\n"
        else:  # shortest_path
            class_idx = data.y.item()
            if hasattr(data, 'query_u') and hasattr(data, 'query_v'):
                query_u = data.query_u.item() if data.query_u.dim() == 0 else data.query_u[0].item()
                query_v = data.query_v.item() if data.query_v.dim() == 0 else data.query_v[0].item()
                examples_str += f"  Query: node {query_u} → node {query_v}\n"
            examples_str += f"  Path length: len{class_idx + 1} (class {class_idx})\n"

        # Show edge list (first 10 edges)
        edges = data.edge_index.t().tolist()
        examples_str += f"  Edges (first 10): {edges[:10]}\n"
        examples_str += "\n"

    examples_str += f"{'='*80}\n"
    return examples_str


def visualize_graph(data, task: str = 'cycle_check', title: str = "Graph") -> Image.Image:
    """
    Visualize a PyG graph using networkx and matplotlib.

    Args:
        data: PyG Data object
        task: 'cycle_check' or 'shortest_path'
        title: Title for the plot

    Returns:
        PIL Image object
    """
    # Create networkx graph from PyG data
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(data.num_nodes))

    # Add edges (convert from PyG edge_index format)
    edge_index = data.edge_index.cpu().numpy()
    edge_list = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edge_list)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Layout - use spring layout for better visualization
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    # Determine node colors and sizes
    node_colors = ['lightblue'] * data.num_nodes
    node_sizes = [500] * data.num_nodes

    # Highlight query nodes for shortest_path task
    if task == 'shortest_path' and hasattr(data, 'query_u') and hasattr(data, 'query_v'):
        query_u = data.query_u.item() if data.query_u.dim() == 0 else data.query_u[0].item()
        query_v = data.query_v.item() if data.query_v.dim() == 0 else data.query_v[0].item()

        # Color query nodes differently
        node_colors[query_u] = '#ff6b6b'  # Red for source
        node_colors[query_v] = '#4ecdc4'  # Cyan for target
        node_sizes[query_u] = 800
        node_sizes[query_v] = 800

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Add title with task-specific information
    if task == 'cycle_check':
        label_str = "Has Cycle" if data.y.item() == 1 else "No Cycle"
        full_title = f"{title}\nLabel: {label_str} | Nodes: {data.num_nodes} | Edges: {data.edge_index.size(1)}"
    else:  # shortest_path
        class_idx = data.y.item()
        if hasattr(data, 'query_u') and hasattr(data, 'query_v'):
            query_u = data.query_u.item() if data.query_u.dim() == 0 else data.query_u[0].item()
            query_v = data.query_v.item() if data.query_v.dim() == 0 else data.query_v[0].item()
            full_title = f"{title}\nQuery: {query_u}→{query_v} | Distance: len{class_idx + 1} | Nodes: {data.num_nodes} | Edges: {data.edge_index.size(1)}"
        else:
            full_title = f"{title}\nDistance: len{class_idx + 1} | Nodes: {data.num_nodes} | Edges: {data.edge_index.size(1)}"

    ax.set_title(full_title, fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_graph_visualizations(dataset, task: str, num_examples: int = 3) -> List[Image.Image]:
    """
    Create visualizations for multiple example graphs.

    Args:
        dataset: PyG dataset
        task: 'cycle_check' or 'shortest_path'
        num_examples: Number of graphs to visualize

    Returns:
        List of PIL Image objects
    """
    images = []
    for i in range(min(num_examples, len(dataset))):
        data = dataset[i]
        img = visualize_graph(data, task=task, title=f"Example Graph {i+1}")
        images.append(img)
    return images
