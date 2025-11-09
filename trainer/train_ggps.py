"""Train GraphGPS on native graphs"""

import os
import argparse
import random
import logging
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything
import wandb

# GraphGPS modules
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'GraphGPS'))
import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig

# graph-token dataset
from graph_data_loader import GraphTokenDataset


class GPSWrapper(nn.Module):
    """
    Wrapper around GraphGym GPS model to add task-specific functionality:
    - Query encoding for shortest_path task
    - Task-aware output head (binary vs multi-class)
    """
    def __init__(self, gps_model, task='cycle_check', num_classes=2):
        super().__init__()
        self.gps_model = gps_model
        self.task = task
        self.num_classes = num_classes

    def forward(self, batch):
        # Add query encoding for shortest_path task
        if self.task == 'shortest_path' and hasattr(batch, 'query_u') and hasattr(batch, 'query_v'):
            from graph_data_loader import add_query_encoding_to_features
            # Add binary positional encoding for query nodes
            # Handle batched query_u and query_v
            if batch.query_u.dim() > 0:
                query_u = batch.query_u[0].item()
                query_v = batch.query_v[0].item()
                batch.x = add_query_encoding_to_features(batch.x, query_u, query_v)

        # Forward through GPS model
        out = self.gps_model(batch)

        return out


def load_config(config_path):
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_cfg_from_config(config_dict):
    """Setup GraphGym config from config dictionary."""
    set_cfg(cfg)

    data_cfg = config_dict.get('data', {})
    cfg.dataset.format = 'custom'
    cfg.dataset.name = data_cfg.get('task', 'cycle_check')
    cfg.dataset.task = 'graph'
    cfg.dataset.task_type = 'classification'

    # model parameters from config file
    for key, value in config_dict.items():
        if key in ['data']:  # Skip our custom section
            continue
        if key in cfg:
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey in cfg[key]:
                        setattr(cfg[key], subkey, subvalue)
            else:
                setattr(cfg, key, value)

    # ensure optimizer params are floats (not strings)
    if hasattr(cfg.optim, 'weight_decay'):
        cfg.optim.weight_decay = float(cfg.optim.weight_decay)
    if hasattr(cfg.optim, 'base_lr'):
        cfg.optim.base_lr = float(cfg.optim.base_lr)
    if hasattr(cfg.optim, 'momentum'):
        cfg.optim.momentum = float(cfg.optim.momentum)

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.num_threads = 4

    return cfg


def accuracy(pred, target, task='cycle_check'):
    """Calculate accuracy for binary or multi-class classification."""
    if task == 'cycle_check':
        # Binary classification with BCE loss
        # pred is (batch_size, 1) or (batch_size,) with logits
        # Apply sigmoid and threshold at 0.5
        pred_binary = (torch.sigmoid(pred.squeeze()) > 0.5).long()
        return (pred_binary == target.long()).float().mean().item()
    else:
        # Multi-class classification (shortest_path)
        # pred is (batch_size, num_classes) with logits
        # target is (batch_size,) with class indices
        pred_classes = pred.argmax(dim=-1)
        return (pred_classes == target).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device, task='cycle_check'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_graphs = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        if isinstance(out, tuple):
            # GraphGym models return (predictions, label)
            pred, _ = out
        else:
            pred = out

        # Handle labels based on task
        target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y

        if task == 'cycle_check':
            # Binary classification with BCE loss - need float targets
            target = target.float()
            pred = pred.squeeze()
        else:
            # Multi-class classification - need long targets
            target = target.long()

        loss = criterion(pred, target)

        loss.backward()
        if cfg.optim.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_acc += accuracy(pred, target, task=task) * batch.num_graphs
        num_graphs += batch.num_graphs

    return total_loss / num_graphs, total_acc / num_graphs


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, task='cycle_check'):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        out = model(batch)

        if isinstance(out, tuple):
            pred, _ = out
        else:
            pred = out

        # Handle labels based on task
        target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y

        if task == 'cycle_check':
            # Binary classification with BCE loss - need float targets
            target = target.float()
            pred = pred.squeeze()
        else:
            # Multi-class classification - need long targets
            target = target.long()

        loss = criterion(pred, target)

        total_loss += loss.item() * batch.num_graphs
        total_acc += accuracy(pred, target, task=task) * batch.num_graphs
        num_graphs += batch.num_graphs

    return total_loss / num_graphs, total_acc / num_graphs


def main(config_dict):
    data_cfg = config_dict.get('data', {})
    wandb_cfg = config_dict.get('wandb', {})

    seed = config_dict.get('seed', 0)
    random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed)

    cfg_obj = setup_cfg_from_config(config_dict)
    logging.basicConfig(level=logging.INFO)

    task = data_cfg.get('task', 'cycle_check')
    algorithm = data_cfg.get('algorithm', 'er')
    graph_token_root = data_cfg.get('graph_token_root', 'graph-token')
    use_split_tasks_dirs = data_cfg.get('use_split_tasks_dirs', True)
    data_fraction = data_cfg.get('data_fraction', 1.0)

    logging.info(f"Loading {task} dataset with algorithm {algorithm}")

    train_dataset = GraphTokenDataset(
        root=graph_token_root,
        task=task,
        algorithm=algorithm,
        split='train',
        use_split_tasks_dirs=use_split_tasks_dirs,
        data_fraction=data_fraction,
        seed=seed,
    )

    try:
        val_dataset = GraphTokenDataset(
            root=graph_token_root,
            task=task,
            algorithm=algorithm,
            split='val',
            use_split_tasks_dirs=use_split_tasks_dirs,
            data_fraction=data_fraction,
            seed=seed,
        )
    except RuntimeError:
        logging.warning("No validation set found, using training set for validation")
        val_dataset = train_dataset

    try:
        test_dataset = GraphTokenDataset(
            root=graph_token_root,
            task=task,
            algorithm=algorithm,
            split='test',
            use_split_tasks_dirs=use_split_tasks_dirs,
            data_fraction=data_fraction,
            seed=seed,
        )
    except RuntimeError:
        logging.warning("No test set found")
        test_dataset = None

    logging.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}" +
                 (f" | Test: {len(test_dataset)}" if test_dataset else ""))

    sample = train_dataset[0]
    logging.info(f"Sample graph: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges, label={sample.y.item()}")

    batch_size = config_dict.get('train', {}).get('batch_size', 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    auto_select_device()

    # Determine number of classes based on task
    if task == 'shortest_path':
        num_classes = 7  # len1 through len7
        dim_out = num_classes
        # For shortest_path, add 2 extra dimensions for query encoding
        dim_in = train_dataset[0].x.size(1) if train_dataset[0].x is not None else 1
        dim_in += 2  # Add query encoding dimensions
    else:  # cycle_check or other binary tasks
        num_classes = 2
        dim_out = 1  # Binary classification with BCE loss (single sigmoid output)
        dim_in = train_dataset[0].x.size(1) if train_dataset[0].x is not None else 1

    # create_model() reads from cfg.share.dim_in and cfg.share.dim_out
    cfg.share.dim_in = dim_in
    cfg.share.dim_out = dim_out

    logging.info(f"Task: {task} | num_classes: {num_classes}")
    logging.info(f"Creating GPSModel with dim_in={dim_in}, dim_out={dim_out}")
    logging.info(f"Config check - cfg.share.dim_in={cfg.share.dim_in}, cfg.share.dim_out={cfg.share.dim_out}")

    # create_model() will read dim_in/dim_out from cfg.share
    gps_model = create_model()

    # Wrap with GPSWrapper to add task-specific functionality
    model = GPSWrapper(gps_model, task=task, num_classes=num_classes)

    logging.info(model)
    num_params = params_count(model)
    logging.info(f'Num parameters: {num_params}')

    optimizer = create_optimizer(
        model.parameters(),
        OptimizerConfig(
            optimizer=cfg.optim.optimizer,
            base_lr=cfg.optim.base_lr,
            weight_decay=cfg.optim.weight_decay,
            momentum=cfg.optim.momentum,
        )
    )

    scheduler = create_scheduler(
        optimizer,
        ExtendedSchedulerConfig(
            scheduler=cfg.optim.scheduler,
            steps=cfg.optim.steps,
            lr_decay=cfg.optim.lr_decay,
            max_epoch=cfg.optim.max_epoch,
            reduce_factor=cfg.optim.reduce_factor,
            schedule_patience=cfg.optim.schedule_patience,
            min_lr=cfg.optim.min_lr,
            num_warmup_epochs=cfg.optim.num_warmup_epochs,
            train_mode=cfg.train.mode,
            eval_period=cfg.train.eval_period,
        )
    )

    # Select loss function based on task
    if task == 'cycle_check':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    use_wandb = wandb_cfg.get('use', False)
    wandb_project = wandb_cfg.get('project', 'graph-token')
    run_name = config_dict.get('run_name', 'gps-cycle-check')

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config=config_dict,
        )
        wandb.watch(model, log='all', log_freq=100)

    out_dir = config_dict.get('out_dir', 'runs_gps')
    epochs = config_dict.get('optim', {}).get('max_epoch', 100)

    os.makedirs(out_dir, exist_ok=True)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, task=task)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, task=task)

        log_dict = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        }

        if use_wandb:
            wandb.log(log_dict)

        logging.info(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {'state_dict': best_state, 'epoch': epoch, 'val_acc': val_acc},
                os.path.join(out_dir, f'best_{run_name}.pt'),
            )

        scheduler.step()

    # test on best model
    if test_dataset and best_state:
        model.load_state_dict(best_state)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device, task=task)
        logging.info(f"Test: {test_loss:.4f}/{test_acc:.4f}")

        if use_wandb:
            wandb.log({'test/loss': test_loss, 'test/acc': test_acc})

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GraphGPS on graph-token tasks')
    parser.add_argument('--config', type=str, default='configs/gps_graph_token.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {args.config}")
    data_cfg = config.get('data', {})
    print(f"Task: {data_cfg.get('task', 'cycle_check')} | Algorithm: {data_cfg.get('algorithm', 'er')}")

    main(config)