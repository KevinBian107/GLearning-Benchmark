"""Train GIN-MPNN on native graph structures"""

import os, argparse, random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.loader import DataLoader
import wandb

from graph_data_loader import GraphTokenDataset


class MPNN(nn.Module):
    """
    Simple Message Passing Neural Network using GIN (Graph Isomorphism Network) layers.
    Supports both cycle_check (binary) and shortest_path (multi-class with query encoding).
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

        # For shortest_path, input features include 2 extra dimensions for query encoding
        actual_in_dim = in_dim + 2 if task == 'shortest_path' else in_dim
        self.node_encoder = nn.Linear(actual_in_dim, hidden_dim)

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

        # Add query encoding for shortest_path task
        if self.task == 'shortest_path' and hasattr(data, 'query_u') and hasattr(data, 'query_v'):
            # Import here to avoid circular dependency
            from graph_data_loader import add_query_encoding_to_features
            # Add binary positional encoding for query nodes
            x = add_query_encoding_to_features(x, data.query_u[0].item(), data.query_v[0].item())

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


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()


def train_one_epoch(model, dl, opt, crit, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in dl:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = crit(logits, batch.y.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, batch.y.squeeze()) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)


@torch.no_grad()
def eval_epoch(model, dl, crit, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in dl:
        batch = batch.to(device)
        logits = model(batch)
        loss = crit(logits, batch.y.squeeze())

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, batch.y.squeeze()) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)


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

    train_dataset = GraphTokenDataset(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithm=dataset_cfg['algorithm'],
        split='train',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
    )
    val_dataset = GraphTokenDataset(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithm=dataset_cfg['algorithm'],
        split='val',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
    )
    test_dataset = GraphTokenDataset(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithm=dataset_cfg['algorithm'],
        split='test',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
    )

    print(f"#train: {len(train_dataset)} | #val: {len(val_dataset)} | #test: {len(test_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training examples found. Did you run the task generator?")
    if len(test_dataset) == 0:
        print(f"[warn] No test files found. Test metrics will be trivial.")

    # sample graph
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"[train] sample: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges, label={sample.y.item()}")

    # data loaders
    train_dl = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    val_dl = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])
    test_dl = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine number of classes based on task
    if dataset_cfg['task'] == 'shortest_path':
        num_classes = model_cfg.get('num_classes', 7)  # Default 7 for len1-len7
    else:  # cycle_check or other binary tasks
        num_classes = model_cfg.get('num_classes', 2)

    model = MPNN(
        in_dim=model_cfg['in_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'],
        pooling=model_cfg['pooling'],
        num_classes=num_classes,
        task=dataset_cfg['task'],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    crit = nn.CrossEntropyLoss()

    # Initialize W&B if enabled
    if wandb_cfg['use']:
        wandb.init(project=wandb_cfg['project'], name=output_cfg['run_name'], config=config)
        wandb.watch(model, log="all", log_freq=100)

    os.makedirs(output_cfg['out_dir'], exist_ok=True)
    best_val, best_state = -1.0, None

    for epoch in range(1, train_cfg['epochs'] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, crit, device)
        va_loss, va_acc = eval_epoch(model, val_dl, crit, device)

        log_dict = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss, "val/acc": va_acc,
            "lr": opt.param_groups[0]["lr"],
        }
        if wandb_cfg['use']:
            wandb.log(log_dict)

        print(f"epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"state_dict": best_state, "config": config},
                os.path.join(output_cfg['out_dir'], f"best_{output_cfg['run_name']}.pt"),
            )

    if best_state:
        model.load_state_dict(best_state)
    te_loss, te_acc = eval_epoch(model.to(device), test_dl, crit, device)
    print(f"TEST loss/acc: {te_loss:.4f}/{te_acc:.4f}")

    if wandb_cfg['use']:
        wandb.log({"test/loss": te_loss, "test/acc": te_acc})
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
