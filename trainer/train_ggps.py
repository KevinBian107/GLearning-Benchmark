"""Train GraphGPS on native graphs"""

import os
import argparse
import random
import logging
import time
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything
from torch_geometric.datasets import ZINC
import wandb

# GraphGPS modules
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'GraphGPS'))
import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig

# graph-token dataset
from graph_data_loader import GraphTokenDataset, AddQueryEncoding, determine_num_classes_pyg
from torch.utils.data import Subset
from trainer.metrics import compute_metrics, aggregate_metrics, format_confusion_matrix, get_loss_function, log_graph_examples, create_graph_visualizations, create_confusion_matrix_heatmap


class GPSWrapper(nn.Module):
    """
    Wrapper around GraphGym GPS model.
    Note: Query encoding for shortest_path is now handled by dataset transform.
    """
    def __init__(self, gps_model, task='cycle_check', num_classes=2):
        super().__init__()
        self.gps_model = gps_model
        self.task = task
        self.num_classes = num_classes

    def forward(self, batch):
        # Forward through GPS model
        # Query encoding is already in batch.x from dataset transform
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
    task = data_cfg.get('task', 'cycle_check')

    cfg.dataset.format = 'custom'
    cfg.dataset.name = task
    cfg.dataset.task = 'graph'

    # Set task type based on task
    if task == 'zinc':
        cfg.dataset.task_type = 'regression'
    else:
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


def train_epoch(model, loader, optimizer, criterion, device, task='cycle_check'):
    """Train for one epoch."""
    model.train()
    all_metrics = []

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        # Convert to float if needed (ZINC has integer atom features)
        if hasattr(batch, 'x') and batch.x is not None and batch.x.dtype != torch.float32:
            batch.x = batch.x.float()

        optimizer.zero_grad()
        out = model(batch)

        if isinstance(out, tuple):
            # GraphGym models return (predictions, label)
            pred, _ = out
        else:
            pred = out

        # Handle labels based on task
        target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y

        if task == 'zinc':
            # Regression task
            target = target.float()
            pred = pred.squeeze(-1)
        elif task == 'cycle_check':
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

        # Compute comprehensive metrics
        batch_metrics = compute_metrics(pred, target, task=task, loss_val=loss.item())
        all_metrics.append(batch_metrics)

    # Aggregate metrics across batches
    return aggregate_metrics(all_metrics)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, task='cycle_check'):
    """Evaluate for one epoch."""
    model.eval()
    all_metrics = []

    for batch in loader:
        batch = batch.to(device)

        # Convert to float if needed (ZINC has integer atom features)
        if hasattr(batch, 'x') and batch.x is not None and batch.x.dtype != torch.float32:
            batch.x = batch.x.float()

        out = model(batch)

        if isinstance(out, tuple):
            pred, _ = out
        else:
            pred = out

        # Handle labels based on task
        target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y

        if task == 'zinc':
            # Regression task
            target = target.float()
            pred = pred.squeeze(-1)
        elif task == 'cycle_check':
            # Binary classification with BCE loss - need float targets
            target = target.float()
            pred = pred.squeeze()
        else:
            # Multi-class classification - need long targets
            target = target.long()

        loss = criterion(pred, target)

        # Compute comprehensive metrics
        batch_metrics = compute_metrics(pred, target, task=task, loss_val=loss.item())
        all_metrics.append(batch_metrics)

    # Aggregate metrics across batches
    return aggregate_metrics(all_metrics)


def main(config_dict):
    data_cfg = config_dict.get('data', {})
    wandb_cfg = config_dict.get('wandb', {})

    seed = config_dict.get('seed', 0)
    random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed)

    cfg_obj = setup_cfg_from_config(config_dict)
    logging.basicConfig(level=logging.INFO)

    # Debug: Verify layer_type is loaded correctly
    print(f"DEBUG: cfg.gt.layer_type = {cfg.gt.layer_type}")

    task = data_cfg.get('task', 'cycle_check')

    # ZINC dataset loading (regression task)
    if task == 'zinc':
        logging.info(f"\n{'='*80}")
        logging.info(f"LOADING DATA - ZINC 12K Regression Dataset")
        logging.info('='*80)
        logging.info(f"Task: Graph-level regression (constrained solubility)")

        zinc_root = data_cfg.get('zinc_root', './data/ZINC')
        subset = data_cfg.get('subset', True)

        train_dataset = ZINC(root=zinc_root, subset=subset, split='train')
        val_dataset = ZINC(root=zinc_root, subset=subset, split='val')
        test_dataset = ZINC(root=zinc_root, subset=subset, split='test')

        logging.info(f"#train: {len(train_dataset)} | #val: {len(val_dataset)} | #test: {len(test_dataset)}")
        logging.info(f"Sample molecule: {train_dataset[0].num_nodes} atoms, {train_dataset[0].num_edges} bonds")
        logging.info(f"Target (first sample): {train_dataset[0].y.item():.4f}")

        graph_images = []  # No algorithm-specific visualization for ZINC

    # Graph-token datasets (classification tasks)
    else:
        # Load data from multiple algorithms for train/val, OOD algorithm for test
        train_algorithms = data_cfg['train_algorithms']
        test_algorithm = data_cfg['test_algorithm']
        graph_token_root = data_cfg.get('graph_token_root', 'graph-token')
        use_split_tasks_dirs = data_cfg.get('use_split_tasks_dirs', True)
        num_graphs = data_cfg.get('num_graphs', None)
        num_pairs_per_graph = data_cfg.get('num_pairs_per_graph', None)

        logging.info(f"Loading {task} dataset with multi-algorithm setup")
        logging.info(f"Train/Val Algorithms: {train_algorithms}")
        logging.info(f"Test Algorithm (OOD): {test_algorithm}")
        logging.info(f"Num Graphs per Algorithm: {num_graphs}")
        logging.info(f"Num Pairs per Graph: {num_pairs_per_graph}")

        # Create transform for shortest_path task (adds query encoding to features)
        pre_transform = AddQueryEncoding() if task == 'shortest_path' else None

        train_dataset = GraphTokenDataset(
            root=graph_token_root,
            task=task,
            algorithm=train_algorithms,
            split='train',
            use_split_tasks_dirs=use_split_tasks_dirs,
            num_graphs=num_graphs,
            num_pairs_per_graph=num_pairs_per_graph,
            seed=seed,
            pre_transform=pre_transform,
        )

        try:
            val_dataset = GraphTokenDataset(
                root=graph_token_root,
                task=task,
                algorithm=train_algorithms,
                split='val',
                use_split_tasks_dirs=use_split_tasks_dirs,
                num_graphs=num_graphs,
                num_pairs_per_graph=num_pairs_per_graph,
                seed=seed,
                pre_transform=pre_transform,
            )
        except RuntimeError:
            logging.warning("No validation set found, using training set for validation")
            val_dataset = train_dataset

        try:
            test_dataset = GraphTokenDataset(
                root=graph_token_root,
                task=task,
                algorithm=[test_algorithm],
                split='test',
                use_split_tasks_dirs=use_split_tasks_dirs,
                num_graphs=num_graphs,
                num_pairs_per_graph=num_pairs_per_graph,
                seed=seed,
                pre_transform=pre_transform,
            )
        except RuntimeError:
            logging.warning("No test set found")
            test_dataset = None

        logging.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}" +
                     (f" | Test: {len(test_dataset)}" if test_dataset else ""))

        # Log example graphs from each algorithm
        print(f"\n{'='*80}")
        print(f"EXAMPLE GRAPHS FROM EACH ALGORITHM")
        print('='*80)

        graph_images = []
        for algo in train_algorithms:
            # Create a small dataset from this algorithm for display
            algo_dataset = GraphTokenDataset(
                root=graph_token_root,
                task=task,
                algorithm=[algo],
                split='train',
                use_split_tasks_dirs=use_split_tasks_dirs,
                num_graphs=1,
                num_pairs_per_graph=1,
                seed=seed,
                pre_transform=pre_transform,
            )
            if len(algo_dataset) > 0:
                print(f"\n[TRAIN - {algo.upper()}]")
                print(log_graph_examples(algo_dataset, task=task, num_examples=1))
                algo_images = create_graph_visualizations(algo_dataset, task=task, num_examples=1)
                graph_images.extend([(f"TRAIN-{algo.upper()}", img) for img in algo_images])

        # Show one example from test (OOD) algorithm
        test_display_dataset = GraphTokenDataset(
            root=graph_token_root,
            task=task,
            algorithm=[test_algorithm],
            split='test',
            use_split_tasks_dirs=use_split_tasks_dirs,
            num_graphs=1,
            num_pairs_per_graph=1,
            seed=seed,
            pre_transform=pre_transform,
        )
        if len(test_display_dataset) > 0:
            print(f"\n[TEST - {test_algorithm.upper()} (OOD)]")
            print(log_graph_examples(test_display_dataset, task=task, num_examples=1))
            test_images = create_graph_visualizations(test_display_dataset, task=task, num_examples=1)
            graph_images.extend([(f"TEST-{test_algorithm.upper()}-OOD", img) for img in test_images])

        print('='*80)
        print(f"\nCreated {len(graph_images)} graph visualizations total")
        print()

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

    # Auto-determine number of classes from ALL data (train, val, test combined)
    from torch.utils.data import ConcatDataset

    if task == 'zinc':
        # ZINC: regression task
        dim_in = train_dataset[0].x.size(1)  # Atom features
        dim_out = 1  # Regression output
        num_classes = 1
    else:
        datasets_to_check = [train_dataset, val_dataset]
        if test_dataset is not None:
            datasets_to_check.append(test_dataset)
        all_dataset = ConcatDataset(datasets_to_check)
        num_classes = determine_num_classes_pyg(all_dataset, task=task)

        # Determine input dimension based on task
        # Note: For shortest_path, query encoding is already added by pre_transform
        if task == 'shortest_path':
            dim_out = num_classes
            # Query encoding already added by transform, so just use actual feature dim
            dim_in = train_dataset[0].x.size(1)
        else:  # cycle_check or other binary tasks
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

    # Store num_params for wandb logging
    model_num_params = num_params

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

    # Select loss function based on task (unified across all models)
    criterion = get_loss_function(task, device)

    use_wandb = wandb_cfg.get('use', False)
    wandb_project = wandb_cfg.get('project', 'graph-token')
    run_name = config_dict.get('run_name', 'gps-cycle-check')

    # Create run name with training algorithms in parentheses (only for graph-token tasks)
    if task == 'zinc':
        wandb_run_name = run_name
    else:
        algo_str = '+'.join(train_algorithms)
        wandb_run_name = f"{run_name} ({algo_str})"

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config_dict,
        )
        wandb.watch(model, log='all', log_freq=100)
        wandb.log({"model/num_parameters": model_num_params})

        # Log graph visualizations to W&B
        print("Logging graph visualizations to W&B...")
        wandb_images = [wandb.Image(img, caption=algo_name) for algo_name, img in graph_images]
        wandb.log({"examples/train_graphs": wandb_images})

    out_dir = config_dict.get('out_dir', 'runs_gps')
    epochs = config_dict.get('optim', {}).get('max_epoch', 100)

    os.makedirs(out_dir, exist_ok=True)

    # Best model tracking (criterion depends on task)
    if task == 'zinc':
        best_val = float('inf')  # Lower MAE is better
    else:
        best_val = 0.0  # Higher accuracy is better
    best_state = None

    # Timing and efficiency tracking
    training_start_time = time.time()
    num_train_graphs = len(train_dataset)
    initial_val_metric = 0.0
    time_to_best = 0.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, task=task)
        val_metrics = eval_epoch(model, val_loader, criterion, device, task=task)

        epoch_duration = time.time() - epoch_start

        # Extract key metrics
        train_loss = train_metrics['loss']
        val_loss = val_metrics['loss']

        # Calculate throughput (graphs per second)
        graphs_per_sec = num_train_graphs / epoch_duration if epoch_duration > 0 else 0

        # GPU memory tracking
        gpu_mem_allocated = 0
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        # Build comprehensive logging dictionary
        log_dict = {
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            "time/epoch_duration": epoch_duration,
            "throughput/graphs_per_sec": graphs_per_sec,
            "memory/gpu_allocated_mb": gpu_mem_allocated,
        }

        # Task-specific metrics
        if task == 'zinc':
            train_mae, val_mae = train_metrics['mae'], val_metrics['mae']
            log_dict.update({
                "train/mae": train_metrics['mae'],
                "train/mse": train_metrics['mse'],
                "train/rmse": train_metrics['rmse'],
                "val/mae": val_metrics['mae'],
                "val/mse": val_metrics['mse'],
                "val/rmse": val_metrics['rmse'],
            })

            logging.info(
                f"Epoch {epoch:03d} | Train: loss={train_loss:.4f}/mae={train_mae:.4f} | "
                f"Val: loss={val_loss:.4f}/mae={val_mae:.4f} | Time: {epoch_duration:.2f}s"
            )

            current_metric = val_mae
            is_best = val_mae < best_val
        else:
            # Classification metrics
            train_acc, val_acc = train_metrics['accuracy'], val_metrics['accuracy']
            log_dict.update({
                'train/acc': train_acc,
                "train/precision": train_metrics.get('precision', train_metrics.get('precision_macro', 0)),
                "train/recall": train_metrics.get('recall', train_metrics.get('recall_macro', 0)),
                "train/f1": train_metrics.get('f1', train_metrics.get('f1_macro', 0)),
                'val/acc': val_acc,
                "val/precision": val_metrics.get('precision', val_metrics.get('precision_macro', 0)),
                "val/recall": val_metrics.get('recall', val_metrics.get('recall_macro', 0)),
                "val/f1": val_metrics.get('f1', val_metrics.get('f1_macro', 0)),
            })

            # Add MSE/MAE for shortest_path
            if task == 'shortest_path':
                log_dict["train/mse"] = train_metrics.get('mse', 0)
                log_dict["train/mae"] = train_metrics.get('mae', 0)
                log_dict["val/mse"] = val_metrics.get('mse', 0)
                log_dict["val/mae"] = val_metrics.get('mae', 0)

                logging.info(
                    f"Epoch {epoch:03d} | "
                    f"Train: {train_loss:.4f}/acc={train_acc:.4f}/mse={train_metrics.get('mse', 0):.4f} | "
                    f"Val: {val_loss:.4f}/acc={val_acc:.4f}/mse={val_metrics.get('mse', 0):.4f} | "
                    f"Time: {epoch_duration:.2f}s"
                )
            else:
                logging.info(
                    f"Epoch {epoch:03d} | "
                    f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                    f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                    f"Time: {epoch_duration:.2f}s"
                )

            current_metric = val_acc
            is_best = val_acc > best_val

        if use_wandb:
            wandb.log(log_dict)

        if is_best:
            best_val = current_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            time_to_best = time.time() - training_start_time
            save_dict = {'state_dict': best_state, 'epoch': epoch}
            if task == 'zinc':
                save_dict['val_mae'] = val_mae
            else:
                save_dict['val_acc'] = val_acc
            torch.save(save_dict, os.path.join(out_dir, f'best_{run_name}.pt'))

        scheduler.step()

    # Log total training time
    total_train_time = time.time() - training_start_time

    # test on best model
    if test_dataset and best_state:
        model.load_state_dict(best_state)
        test_metrics = eval_epoch(model, test_loader, criterion, device, task=task)
        test_loss = test_metrics['loss']

        # Print test results
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Loss: {test_loss:.4f}")

        if task == 'zinc':
            # Regression metrics
            test_mae = test_metrics['mae']
            test_mse = test_metrics['mse']
            test_rmse = test_metrics['rmse']
            print(f"MAE: {test_mae:.4f}")
            print(f"MSE: {test_mse:.4f}")
            print(f"RMSE: {test_rmse:.4f}")
        else:
            # Classification metrics
            test_acc = test_metrics['accuracy']
            print(f"Accuracy: {test_acc:.4f}")
            print(f"Precision: {test_metrics.get('precision', test_metrics.get('precision_macro', 0)):.4f}")
            print(f"Recall: {test_metrics.get('recall', test_metrics.get('recall_macro', 0)):.4f}")
            print(f"F1 Score: {test_metrics.get('f1', test_metrics.get('f1_macro', 0)):.4f}")

            if task == 'shortest_path':
                print(f"MSE: {test_metrics.get('mse', 0):.4f}")
                print(f"MAE: {test_metrics.get('mae', 0):.4f}")

            # Print confusion matrix
            if 'confusion_matrix' in test_metrics:
                print("\n" + format_confusion_matrix(test_metrics['confusion_matrix'], task=task))

        print(f"\nTotal training time: {total_train_time:.2f}s ({total_train_time/60:.2f}min)")
        print(f"Time to best validation: {time_to_best:.2f}s ({time_to_best/60:.2f}min)")
        print("="*80)

        if use_wandb:
            test_log = {
                'test/loss': test_loss,
                "time/total_train_time": total_train_time,
                "time/time_to_best_val": time_to_best,
            }

            if task == 'zinc':
                # Regression metrics
                test_log.update({
                    'test/mae': test_metrics['mae'],
                    'test/mse': test_metrics['mse'],
                    'test/rmse': test_metrics['rmse'],
                })
            else:
                # Classification metrics
                test_log.update({
                    'test/acc': test_metrics['accuracy'],
                    "test/precision": test_metrics.get('precision', test_metrics.get('precision_macro', 0)),
                    "test/recall": test_metrics.get('recall', test_metrics.get('recall_macro', 0)),
                    "test/f1": test_metrics.get('f1', test_metrics.get('f1_macro', 0)),
                })

                if task == 'shortest_path':
                    test_log["test/mse"] = test_metrics.get('mse', 0)
                    test_log["test/mae"] = test_metrics.get('mae', 0)

            wandb.log(test_log)

            # Log confusion matrix as table
            if 'confusion_matrix' in test_metrics:
                cm = test_metrics['confusion_matrix']
                if task == 'cycle_check':
                    labels = ['No', 'Yes']
                else:
                    labels = [f'len{i+1}' for i in range(7)]

                # Create confusion matrix heatmap
                cm_heatmap = create_confusion_matrix_heatmap(cm, task=task, title="Test Confusion Matrix")
                wandb.log({"test/confusion_matrix_heatmap": wandb.Image(cm_heatmap, caption="Confusion Matrix")})

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