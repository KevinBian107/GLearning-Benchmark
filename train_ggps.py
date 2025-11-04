"""Train GraphGPS on graph-token generated graphs"""

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

# Import GraphGPS modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GraphGPS'))
import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig

# Import our custom dataset
from graph_token_dataset import GraphTokenDataset


def setup_cfg_from_args(args):
    """Setup GraphGym config from command-line arguments."""
    # Load base config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Update config with command-line args
    set_cfg(cfg)

    # Set dataset parameters
    cfg.dataset.format = 'custom'
    cfg.dataset.name = args.task
    cfg.dataset.task = 'graph'
    cfg.dataset.task_type = 'classification'

    # Set model parameters from config file
    for key, value in config_dict.items():
        if key in cfg:
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey in cfg[key]:
                        setattr(cfg[key], subkey, subvalue)
            else:
                setattr(cfg, key, value)

    # Override with command-line args
    cfg.seed = args.seed
    cfg.wandb.use = args.use_wandb
    cfg.wandb.project = args.wandb_project
    cfg.optim.max_epoch = args.epochs
    cfg.optim.base_lr = float(args.lr)
    cfg.train.batch_size = args.batch_size
    cfg.out_dir = args.out_dir
    cfg.metric_best = 'accuracy'
    cfg.metric_agg = 'argmax'

    # Ensure optimizer params are floats (not strings)
    cfg.optim.weight_decay = float(cfg.optim.weight_decay)
    cfg.optim.base_lr = float(cfg.optim.base_lr)
    if hasattr(cfg.optim, 'momentum'):
        cfg.optim.momentum = float(cfg.optim.momentum)

    # Set device
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.num_threads = 4

    return cfg


def accuracy(pred, target):
    """Calculate accuracy for binary classification with BCE loss."""
    # pred is (batch_size, 1) or (batch_size,) with logits
    # target is (batch_size,) with 0/1
    # Apply sigmoid and threshold at 0.5
    pred_binary = (torch.sigmoid(pred.squeeze()) > 0.5).long()
    return (pred_binary == target.long()).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_graphs = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch)

        # GraphGym models return (logits, predictions) tuple
        # We want the logits (first element) for training
        if isinstance(out, tuple):
            pred, _ = out
        else:
            pred = out

        # Get labels and convert to float for BCE loss
        target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
        target = target.float()

        # Squeeze pred to match target shape
        pred = pred.squeeze()

        # Calculate loss (both pred and target should be (batch_size,))
        loss = criterion(pred, target)

        # Backward pass
        loss.backward()
        if cfg.optim.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        total_acc += accuracy(pred, target) * batch.num_graphs
        num_graphs += batch.num_graphs

    return total_loss / num_graphs, total_acc / num_graphs


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        out = model(batch)

        # GraphGym models return (logits, predictions) tuple
        # We want the logits (first element)
        if isinstance(out, tuple):
            pred, _ = out
        else:
            pred = out

        # Get labels and convert to float for BCE loss
        target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
        target = target.float()

        # Squeeze pred to match target shape
        pred = pred.squeeze()

        # Calculate loss (both pred and target should be (batch_size,))
        loss = criterion(pred, target)

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        total_acc += accuracy(pred, target) * batch.num_graphs
        num_graphs += batch.num_graphs

    return total_loss / num_graphs, total_acc / num_graphs


def main(args):
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_everything(args.seed)

    # Setup config
    cfg_obj = setup_cfg_from_args(args)
    logging.basicConfig(level=logging.INFO)

    # Load datasets
    logging.info(f"Loading {args.task} dataset with algorithm {args.algorithm}")

    train_dataset = GraphTokenDataset(
        root=args.graph_token_root,
        task=args.task,
        algorithm=args.algorithm,
        split='train',
        use_split_tasks_dirs=True,
    )

    # Try to load validation set, fallback to using train if not available
    try:
        val_dataset = GraphTokenDataset(
            root=args.graph_token_root,
            task=args.task,
            algorithm=args.algorithm,
            split='val',
            use_split_tasks_dirs=True,
        )
    except RuntimeError:
        logging.warning("No validation set found, using training set for validation")
        val_dataset = train_dataset

    try:
        test_dataset = GraphTokenDataset(
            root=args.graph_token_root,
            task=args.task,
            algorithm=args.algorithm,
            split='test',
            use_split_tasks_dirs=True,
        )
    except RuntimeError:
        logging.warning("No test set found")
        test_dataset = None

    logging.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}" +
                 (f" | Test: {len(test_dataset)}" if test_dataset else ""))

    # Show sample
    sample = train_dataset[0]
    logging.info(f"Sample graph: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges, label={sample.y.item()}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    auto_select_device()

    # Get input/output dimensions from dataset
    dim_in = train_dataset[0].x.size(1) if train_dataset[0].x is not None else 1
    dim_out = 1  # Binary classification with BCE loss (single sigmoid output)

    # Set dimensions in config (required for create_model)
    # create_model() reads from cfg.share.dim_in and cfg.share.dim_out
    cfg.share.dim_in = dim_in
    cfg.share.dim_out = dim_out

    logging.info(f"Creating GPSModel with dim_in={dim_in}, dim_out={dim_out} (BCE loss)")
    logging.info(f"Config check - cfg.share.dim_in={cfg.share.dim_in}, cfg.share.dim_out={cfg.share.dim_out}")

    # create_model() will read dim_in/dim_out from cfg.share
    model = create_model()

    # Log model info
    logging.info(model)
    num_params = params_count(model)
    logging.info(f'Num parameters: {num_params}')

    # Setup optimizer and scheduler
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

    # Setup loss function
    # Use BCE for binary classification with single sigmoid output
    criterion = nn.BCEWithLogitsLoss()

    # Setup wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )
        wandb.watch(model, log='all', log_freq=100)

    # Training loop
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        # Log
        log_dict = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        }

        if args.use_wandb:
            wandb.log(log_dict)

        logging.info(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {'state_dict': best_state, 'epoch': epoch, 'val_acc': val_acc},
                os.path.join(args.out_dir, f'best_{args.run_name}.pt'),
            )

        # Step scheduler
        scheduler.step()

    # Test on best model
    if test_dataset and best_state:
        model.load_state_dict(best_state)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        logging.info(f"Test: {test_loss:.4f}/{test_acc:.4f}")

        if args.use_wandb:
            wandb.log({'test/loss': test_loss, 'test/acc': test_acc})

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GraphGPS on graph-token tasks')

    # Dataset args
    parser.add_argument('--graph_token_root', type=str, default='graph-token',
                        help='Path to graph-token repository')
    parser.add_argument('--task', type=str, default='cycle_check',
                        choices=['cycle_check', 'edge_existence', 'node_degree',
                                'node_count', 'edge_count', 'connected_nodes'],
                        help='Task name')
    parser.add_argument('--algorithm', type=str, default='er',
                        help='Graph generation algorithm (er, ba, sbm, etc.)')

    # Model args
    parser.add_argument('--config', type=str, default='configs/gps_graph_token.yaml',
                        help='Path to config file')

    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Logging args
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='graph-token',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default='gps-cycle-check',
                        help='Run name for logging')
    parser.add_argument('--out_dir', type=str, default='runs_gps',
                        help='Output directory for checkpoints')

    args = parser.parse_args()
    main(args)
