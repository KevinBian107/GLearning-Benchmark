"""Train GIN-MPNN on native graph structures"""

import os, argparse, random, time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.loader import DataLoader
import wandb

from graph_data_loader import GraphTokenDataset, AddQueryEncoding
from trainer.metrics import compute_metrics, aggregate_metrics, format_confusion_matrix, get_loss_function, log_graph_examples, create_graph_visualizations


class MPNN(nn.Module):
    """
    Simple Message Passing Neural Network using GIN (Graph Isomorphism Network) layers.
    Supports both cycle_check (binary) and shortest_path (multi-class with query encoding).
    Note: For shortest_path, query encoding is added by dataset transform, not in forward pass.
    """
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        pooling: str = 'mean',
        num_classes: int = 2,
        task: str = 'cycle_check',
    ):
        super().__init__()
        self.pooling = pooling
        self.task = task

        # in_dim should already include query encoding if task is shortest_path
        # (added by dataset transform)
        self.node_encoder = nn.Linear(in_dim, hidden_dim)

        # GIN convolution layers with MLP
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Task-aware classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Query encoding is already in data.x from dataset transform (for shortest_path)
        # encode node features
        x = self.node_encoder(x)

        # message passing layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # graph-level pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)

        return self.classifier(x)


def train_one_epoch(model, dl, opt, crit, device, task='cycle_check'):
    model.train()
    all_metrics = []
    for batch in dl:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        labels = batch.y.squeeze()
        loss = crit(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Compute comprehensive metrics
        batch_metrics = compute_metrics(logits, labels, task=task, loss_val=loss.item())
        all_metrics.append(batch_metrics)

    # Aggregate metrics across batches
    return aggregate_metrics(all_metrics)


@torch.no_grad()
def eval_epoch(model, dl, crit, device, task='cycle_check'):
    model.eval()
    all_metrics = []
    for batch in dl:
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.y.squeeze()
        loss = crit(logits, labels)

        # Compute comprehensive metrics
        batch_metrics = compute_metrics(logits, labels, task=task, loss_val=loss.item())
        all_metrics.append(batch_metrics)

    # Aggregate metrics across batches
    return aggregate_metrics(all_metrics)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    train_cfg = config['train']
    output_cfg = config['output']
    wandb_cfg = config['wandb']

    random.seed(train_cfg['seed'])
    torch.manual_seed(train_cfg['seed'])

    data_fraction = dataset_cfg.get('data_fraction', 1.0)
    seed = train_cfg['seed']

    # Create transform for shortest_path task (adds query encoding to features)
    pre_transform = AddQueryEncoding() if dataset_cfg['task'] == 'shortest_path' else None

    train_dataset = GraphTokenDataset(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithm=dataset_cfg['algorithm'],
        split='train',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
        data_fraction=data_fraction,
        seed=seed,
        pre_transform=pre_transform,
    )
    val_dataset = GraphTokenDataset(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithm=dataset_cfg['algorithm'],
        split='val',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
        data_fraction=data_fraction,
        seed=seed,
        pre_transform=pre_transform,
    )
    test_dataset = GraphTokenDataset(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithm=dataset_cfg['algorithm'],
        split='test',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
        data_fraction=data_fraction,
        seed=seed,
        pre_transform=pre_transform,
    )

    print(f"#train: {len(train_dataset)} | #val: {len(val_dataset)} | #test: {len(test_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training examples found. Did you run the task generator?")
    if len(test_dataset) == 0:
        print(f"[warn] No test files found. Test metrics will be trivial.")

    # Log example graphs (text)
    print(log_graph_examples(train_dataset, task=dataset_cfg['task'], num_examples=2))

    # Create and save graph visualizations
    print("\nCreating graph visualizations...")
    graph_images = create_graph_visualizations(train_dataset, task=dataset_cfg['task'], num_examples=3)
    print(f"Created {len(graph_images)} graph visualizations")

    # data loaders
    train_dl = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    val_dl = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])
    test_dl = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine number of classes and input dimension based on task
    # Note: For shortest_path, query encoding is already added by pre_transform
    if dataset_cfg['task'] == 'shortest_path':
        num_classes = model_cfg.get('num_classes', 7)  # Default 7 for len1-len7
        # Query encoding already added by transform, so use actual feature dim
        in_dim = train_dataset[0].x.size(1)
    else:  # cycle_check or other binary tasks
        num_classes = model_cfg.get('num_classes', 2)
        in_dim = model_cfg['in_dim']

    model = MPNN(
        in_dim=in_dim,
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'],
        pooling=model_cfg['pooling'],
        num_classes=num_classes,
        task=dataset_cfg['task'],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])

    # Select loss function based on task
    crit = get_loss_function(dataset_cfg['task'], device)

    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Initialize W&B if enabled
    if wandb_cfg['use']:
        wandb.init(project=wandb_cfg['project'], name=output_cfg['run_name'], config=config)
        wandb.watch(model, log="all", log_freq=100)
        wandb.log({"model/num_parameters": num_params})

        # Log graph visualizations to W&B
        print("Logging graph visualizations to W&B...")
        wandb_images = [wandb.Image(img, caption=f"Example Graph {i+1}") for i, img in enumerate(graph_images)]
        wandb.log({"examples/train_graphs": wandb_images})

    os.makedirs(output_cfg['out_dir'], exist_ok=True)
    best_val, best_state = -1.0, None

    # Timing and efficiency tracking
    training_start_time = time.time()
    num_train_graphs = len(train_dataset)
    initial_val_acc = 0.0
    time_to_best = 0.0

    for epoch in range(1, train_cfg['epochs'] + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_dl, opt, crit, device, task=dataset_cfg['task'])
        val_metrics = eval_epoch(model, val_dl, crit, device, task=dataset_cfg['task'])

        epoch_duration = time.time() - epoch_start

        # Extract key metrics
        tr_loss, tr_acc = train_metrics['loss'], train_metrics['accuracy']
        va_loss, va_acc = val_metrics['loss'], val_metrics['accuracy']

        # Calculate throughput (graphs per second)
        graphs_per_sec = num_train_graphs / epoch_duration if epoch_duration > 0 else 0

        # GPU memory tracking
        gpu_mem_allocated = 0
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        # Calculate efficiency: time per 1% accuracy gain
        acc_gain = (va_acc - initial_val_acc) * 100  # Convert to percentage
        elapsed_time = time.time() - training_start_time
        time_per_1pct_acc = elapsed_time / acc_gain if acc_gain > 0 else 0

        # Build comprehensive logging dictionary
        log_dict = {
            "epoch": epoch,
            "train/loss": tr_loss,
            "train/acc": tr_acc,
            "train/precision": train_metrics.get('precision', train_metrics.get('precision_macro', 0)),
            "train/recall": train_metrics.get('recall', train_metrics.get('recall_macro', 0)),
            "train/f1": train_metrics.get('f1', train_metrics.get('f1_macro', 0)),
            "val/loss": va_loss,
            "val/acc": va_acc,
            "val/precision": val_metrics.get('precision', val_metrics.get('precision_macro', 0)),
            "val/recall": val_metrics.get('recall', val_metrics.get('recall_macro', 0)),
            "val/f1": val_metrics.get('f1', val_metrics.get('f1_macro', 0)),
            "lr": opt.param_groups[0]["lr"],
            "time/epoch_duration": epoch_duration,
            "throughput/graphs_per_sec": graphs_per_sec,
            "memory/gpu_allocated_mb": gpu_mem_allocated,
            "efficiency/time_per_1pct_acc": time_per_1pct_acc,
        }

        # Add MSE/MAE for shortest_path
        if dataset_cfg['task'] == 'shortest_path':
            log_dict["train/mse"] = train_metrics.get('mse', 0)
            log_dict["train/mae"] = train_metrics.get('mae', 0)
            log_dict["val/mse"] = val_metrics.get('mse', 0)
            log_dict["val/mae"] = val_metrics.get('mae', 0)

        if wandb_cfg['use']:
            wandb.log(log_dict)

        # Print progress
        if dataset_cfg['task'] == 'shortest_path':
            print(f"epoch {epoch:03d} | train {tr_loss:.4f}/acc={tr_acc:.4f}/mse={train_metrics.get('mse', 0):.4f} | "
                  f"val {va_loss:.4f}/acc={va_acc:.4f}/mse={val_metrics.get('mse', 0):.4f} | time {epoch_duration:.2f}s")
        else:
            print(f"epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | time {epoch_duration:.2f}s")

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            time_to_best = time.time() - training_start_time
            torch.save(
                {"state_dict": best_state, "config": config},
                os.path.join(output_cfg['out_dir'], f"best_{output_cfg['run_name']}.pt"),
            )

    # Log total training time
    total_train_time = time.time() - training_start_time

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = eval_epoch(model.to(device), test_dl, crit, device, task=dataset_cfg['task'])
    te_loss, te_acc = test_metrics['loss'], test_metrics['accuracy']

    # Print test results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Loss: {te_loss:.4f}")
    print(f"Accuracy: {te_acc:.4f}")
    print(f"Precision: {test_metrics.get('precision', test_metrics.get('precision_macro', 0)):.4f}")
    print(f"Recall: {test_metrics.get('recall', test_metrics.get('recall_macro', 0)):.4f}")
    print(f"F1 Score: {test_metrics.get('f1', test_metrics.get('f1_macro', 0)):.4f}")

    if dataset_cfg['task'] == 'shortest_path':
        print(f"MSE: {test_metrics.get('mse', 0):.4f}")
        print(f"MAE: {test_metrics.get('mae', 0):.4f}")

    # Print confusion matrix
    if 'confusion_matrix' in test_metrics:
        print("\n" + format_confusion_matrix(test_metrics['confusion_matrix'], task=dataset_cfg['task']))

    print(f"\nTotal training time: {total_train_time:.2f}s ({total_train_time/60:.2f}min)")
    print(f"Time to best validation: {time_to_best:.2f}s ({time_to_best/60:.2f}min)")
    print("="*80)

    if wandb_cfg['use']:
        test_log = {
            "test/loss": te_loss,
            "test/acc": te_acc,
            "test/precision": test_metrics.get('precision', test_metrics.get('precision_macro', 0)),
            "test/recall": test_metrics.get('recall', test_metrics.get('recall_macro', 0)),
            "test/f1": test_metrics.get('f1', test_metrics.get('f1_macro', 0)),
            "time/total_train_time": total_train_time,
            "time/time_to_best_val": time_to_best,
        }
        if dataset_cfg['task'] == 'shortest_path':
            test_log["test/mse"] = test_metrics.get('mse', 0)
            test_log["test/mae"] = test_metrics.get('mae', 0)

        wandb.log(test_log)

        # Log confusion matrix as table
        if 'confusion_matrix' in test_metrics:
            cm = test_metrics['confusion_matrix']
            if dataset_cfg['task'] == 'cycle_check':
                labels = ['No', 'Yes']
            else:
                labels = [f'len{i+1}' for i in range(7)]

            # Create confusion matrix table for W&B
            cm_data = []
            for i, true_label in enumerate(labels):
                row = [true_label] + cm[i].tolist()
                cm_data.append(row)

            cm_table = wandb.Table(
                columns=["True/Pred"] + labels,
                data=cm_data
            )
            wandb.log({"test/confusion_matrix": cm_table})

        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train MPNN on graph-token tasks")
    ap.add_argument("--config", type=str, default="configs/mpnn_graph_token.yaml",
                    help="Path to YAML config file")
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Task: {config['dataset']['task']} | Algorithm: {config['dataset']['algorithm']}")

    main(config)
