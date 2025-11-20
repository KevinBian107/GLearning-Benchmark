"""Train mini-transformer on tokenized graph"""

import os, argparse, random, time
import glob
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from graph_data_loader import (
    SPECIAL,
    load_examples_multi_algorithm,
    determine_num_classes,
    build_vocab_from_texts,
    TokenDataset,
    collate,
)
from trainer.metrics import compute_metrics, aggregate_metrics, format_confusion_matrix, get_loss_function, create_confusion_matrix_heatmap


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        nlayers: int = 4,
        d_ff: int = 512,
        p_drop: float = 0.1,
        max_pos: int = 4096,
        num_classes: int = 2,
        use_query_nodes: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_query_nodes = use_query_nodes

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_pos, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=p_drop,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)

        # Classifier takes 3*d_model if using query nodes, else d_model
        cls_input_dim = 3 * d_model if use_query_nodes else d_model
        self.cls = nn.Linear(cls_input_dim, num_classes)

        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos.weight, std=0.02)
        nn.init.trunc_normal_(self.cls.weight, std=0.02)
        nn.init.zeros_(self.cls.bias)

    def extract_query_nodes(self, x: torch.Tensor, h: torch.Tensor, vocab: dict) -> tuple:
        """
        Extract embeddings for query nodes u and v from shortest_path task.

        Args:
            x: Input token IDs (batch, seq_len)
            h: Transformer hidden states (batch, seq_len, d_model)
            vocab: Vocabulary dict to find <q> token

        Returns:
            (u_emb, v_emb): Query node embeddings (batch, d_model) each
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize with zeros (fallback if we can't find query)
        u_emb = torch.zeros(batch_size, self.d_model, device=device)
        v_emb = torch.zeros(batch_size, self.d_model, device=device)

        # Find <q> token ID
        q_token_id = vocab.get('<q>', -1)
        if q_token_id == -1:
            return u_emb, v_emb

        # For each sample in batch, find query nodes
        for b in range(batch_size):
            # Find position of <q> token
            q_positions = (x[b] == q_token_id).nonzero(as_tuple=True)[0]

            if len(q_positions) == 0:
                continue

            q_pos = q_positions[0].item()

            # Query format: <q> shortest_distance u v <p>
            # So u is at q_pos + 2, v is at q_pos + 3
            if q_pos + 3 < x.size(1):
                # Use embeddings at query positions
                u_emb[b] = h[b, q_pos + 2]
                v_emb[b] = h[b, q_pos + 3]

        return u_emb, v_emb

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, vocab: dict = None) -> torch.Tensor:
        B, L = x.size()
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embed(x) + self.pos(pos_ids)
        key_pad = ~attn_mask  # True where to mask
        h = self.enc(h, src_key_padding_mask=key_pad)

        # Get <bos> embedding
        BOS_ID = SPECIAL.index("<bos>")
        if (x[:, 0] == BOS_ID).all():
            bos_emb = h[:, 0]
        else:
            lens = attn_mask.sum(-1, keepdim=True).clamp(min=1)
            bos_emb = (h * attn_mask.unsqueeze(-1)).sum(1) / lens

        # Extract query node embeddings if enabled and vocab provided
        if self.use_query_nodes and vocab is not None:
            u_emb, v_emb = self.extract_query_nodes(x, h, vocab)
            # Apply LayerNorm to each embedding separately, then concatenate
            bos_normed = self.norm(bos_emb)
            u_normed = self.norm(u_emb)
            v_normed = self.norm(v_emb)
            pooled = torch.cat([bos_normed, u_normed, v_normed], dim=-1)
        else:
            pooled = self.norm(bos_emb)

        return self.cls(pooled)

def train_one_epoch(model, dl, opt, crit, device, vocab=None, task='cycle_check'):
    model.train()
    all_metrics = []
    for X, A, Y in dl:
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(X, A, vocab=vocab)
        loss = crit(logits, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Compute comprehensive metrics
        batch_metrics = compute_metrics(logits, Y, task=task, loss_val=loss.item())
        all_metrics.append(batch_metrics)

    # Aggregate metrics across batches
    return aggregate_metrics(all_metrics)

@torch.no_grad()
def eval_epoch(model, dl, crit, device, vocab=None, task='cycle_check'):
    model.eval()
    all_metrics = []
    for X, A, Y in dl:
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        logits = model(X, A, vocab=vocab)
        loss = crit(logits, Y)

        # Compute comprehensive metrics
        batch_metrics = compute_metrics(logits, Y, task=task, loss_val=loss.item())
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

    # Load data from multiple algorithms for train/val, OOD algorithm for test
    train_algorithms = dataset_cfg['train_algorithms']
    test_algorithm = dataset_cfg['test_algorithm']
    num_graphs = dataset_cfg.get('num_graphs', None)
    num_pairs_per_graph = dataset_cfg.get('num_pairs_per_graph', None)
    seed = train_cfg['seed']

    print(f"\n{'='*80}")
    print(f"LOADING DATA - Multi-Algorithm Setup")
    print('='*80)
    print(f"Train/Val Algorithms: {train_algorithms}")
    print(f"Test Algorithm (OOD): {test_algorithm}")
    print(f"Num Graphs per Algorithm: {num_graphs}")
    print(f"Num Pairs per Graph: {num_pairs_per_graph}")
    print()

    train_ex = load_examples_multi_algorithm(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithms=train_algorithms,
        split='train',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
        seed=seed,
        num_graphs=num_graphs,
        num_pairs_per_graph=num_pairs_per_graph
    )

    val_ex = load_examples_multi_algorithm(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithms=train_algorithms,
        split='val',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
        seed=seed,
        num_graphs=num_graphs,
        num_pairs_per_graph=num_pairs_per_graph
    )

    # Test uses OOD algorithm
    test_ex = load_examples_multi_algorithm(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithms=[test_algorithm],
        split='test',
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
        seed=seed,
        num_graphs=num_graphs,
        num_pairs_per_graph=num_pairs_per_graph
    )

    # Show one example from each training algorithm
    print(f"\n{'='*80}")
    print(f"EXAMPLE GRAPHS FROM EACH ALGORITHM")
    print('='*80)

    for algo in train_algorithms:
        # Load a small sample from this algorithm to show
        if dataset_cfg['use_split_tasks_dirs']:
            base = os.path.join(dataset_cfg['graph_token_root'], 'tasks_train', dataset_cfg['task'], algo)
        else:
            base = os.path.join(dataset_cfg['graph_token_root'], 'tasks', dataset_cfg['task'], algo)

        split_dir = os.path.join(base, 'train')
        path_glob = os.path.join(split_dir, '*.json')
        files = sorted(glob.glob(path_glob))
        if files and len(files) > 0:
            # Load just one file from this algorithm
            from graph_data_loader import load_examples
            algo_examples = load_examples(files[0], task=dataset_cfg['task'], num_graphs=1, num_pairs_per_graph=1)
            if algo_examples:
                ex = algo_examples[0]
                toks = ex["text"].split()
                print(f"\n[TRAIN - {algo.upper()}] len={len(toks)}, label={ex.get('label','?')}")
                print(" ".join(toks[:40]) + (" ..." if len(toks) > 40 else ""))

    # Show one example from test (OOD) algorithm
    if dataset_cfg['use_split_tasks_dirs']:
        base = os.path.join(dataset_cfg['graph_token_root'], 'tasks_test', dataset_cfg['task'], test_algorithm)
    else:
        base = os.path.join(dataset_cfg['graph_token_root'], 'tasks', dataset_cfg['task'], test_algorithm)

    split_dir = os.path.join(base, 'test')
    path_glob = os.path.join(split_dir, '*.json')
    files = sorted(glob.glob(path_glob))
    if files and len(files) > 0:
        from graph_data_loader import load_examples
        test_examples_display = load_examples(files[0], task=dataset_cfg['task'], num_graphs=1, num_pairs_per_graph=1)
        if test_examples_display:
            ex = test_examples_display[0]
            toks = ex["text"].split()
            print(f"\n[TEST - {test_algorithm.upper()} (OOD)] len={len(toks)}, label={ex.get('label','?')}")
            print(" ".join(toks[:40]) + (" ..." if len(toks) > 40 else ""))

    print('='*80)
    print()

    print(f"#train: {len(train_ex)} | #val: {len(val_ex)} | #test: {len(test_ex)}")
    if len(train_ex) == 0:
        raise RuntimeError(f"No training examples found at {train_glob}. Did you run the task generator?")
    if len(test_ex) == 0:
        print(f"[warn] No test files at {test_glob}. Test metrics will be trivial.")

    vocab, _ = build_vocab_from_texts([e["text"] for e in train_ex], max_tokens=dataset_cfg['max_vocab'])
    pad_id = vocab["<pad>"]

    train_ds = TokenDataset(train_ex, vocab, dataset_cfg['max_len'])
    val_ds = TokenDataset(val_ex,   vocab, dataset_cfg['max_len'])
    test_ds = TokenDataset(test_ex,  vocab, dataset_cfg['max_len'])

    coll = lambda b: collate(b, pad_id)
    train_dl = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True,  num_workers=train_cfg['num_workers'], collate_fn=coll)
    val_dl = DataLoader(val_ds,   batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], collate_fn=coll)
    test_dl = DataLoader(test_ds,  batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], collate_fn=coll)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-determine number of classes from ALL data (train, val, test combined)
    all_examples = train_ex + val_ex + test_ex
    num_classes = determine_num_classes(all_examples, task=dataset_cfg['task'])

    # Determine if we should use query node embeddings
    if dataset_cfg['task'] == 'shortest_path':
        use_query_nodes = True  # Enable query node embeddings for shortest_path
    else:  # cycle_check or other binary tasks
        use_query_nodes = False  # Not needed for global properties

    model = SimpleTransformer(
        vocab_size=len(vocab),
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        nlayers=model_cfg['nlayers'],
        d_ff=model_cfg['d_ff'],
        p_drop=model_cfg['dropout'],
        max_pos=model_cfg['max_pos'],
        num_classes=num_classes,
        use_query_nodes=use_query_nodes,
    ).to(device)

    if use_query_nodes:
        print(f"Using query node embeddings (classifier input: {3 * model_cfg['d_model']})")
    else:
        print(f"Using single embedding (classifier input: {model_cfg['d_model']})")

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])

    # Select loss function based on task
    crit = get_loss_function(dataset_cfg['task'], device)

    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    if wandb_cfg['use']:
        wandb.init(project=wandb_cfg['project'], name=output_cfg['run_name'], config=config)
        wandb.watch(model, log="all", log_freq=100)
        wandb.log({"model/num_parameters": num_params})

    os.makedirs(output_cfg['out_dir'], exist_ok=True)
    best_val, best_state = -1.0, None

    # Timing and efficiency tracking
    training_start_time = time.time()
    num_train_graphs = len(train_ds)
    initial_val_acc = 0.0
    time_to_best = 0.0

    for epoch in range(1, train_cfg['epochs'] + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_dl, opt, crit, device, vocab=vocab, task=dataset_cfg['task'])
        val_metrics = eval_epoch(model, val_dl, crit, device, vocab=vocab, task=dataset_cfg['task'])

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
                {"state_dict": best_state, "vocab": vocab, "config": config},
                os.path.join(output_cfg['out_dir'], f"best_{output_cfg['run_name']}.pt"),
            )

    # Log total training time
    total_train_time = time.time() - training_start_time

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = eval_epoch(model.to(device), test_dl, crit, device, vocab=vocab, task=dataset_cfg['task'])
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

        # Log confusion matrix as table and heatmap
        if 'confusion_matrix' in test_metrics:
            cm = test_metrics['confusion_matrix']
            if dataset_cfg['task'] == 'cycle_check':
                labels = ['No', 'Yes']
            else:
                labels = [f'len{i+1}' for i in range(7)]

            # Create confusion matrix heatmap
            cm_heatmap = create_confusion_matrix_heatmap(cm, task=dataset_cfg['task'], title="Test Confusion Matrix")
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

        wandb.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Graph Tokenization + Transformer on graph-token tasks")
    ap.add_argument("--config", type=str, default="configs/gtt_graph_token.yaml",
                    help="Path to YAML config file")
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Task: {config['dataset']['task']} | Algorithm: {config['dataset']['algorithm']}")

    main(config)