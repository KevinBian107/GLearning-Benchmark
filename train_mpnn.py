"""Train vanilla MPNN on native graph structures"""

import os, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.loader import DataLoader
import wandb

from graph_token_dataset import GraphTokenDataset


class MPNN(nn.Module):
    """
    Simple Message Passing Neural Network using GIN (Graph Isomorphism Network) layers.

    Architecture:
        1. Node encoder: Embed 1D constant features to hidden_dim
        2. Message passing: L layers of GIN convolution
        3. Graph pooling: Aggregate node embeddings to graph-level
        4. Classifier: Linear layer for binary prediction
    """
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        pooling: str = 'mean',
    ):
        super().__init__()
        self.pooling = pooling
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

        # Classifier head for binary prediction (2 classes like GTT)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode node features
        x = self.node_encoder(x)

        # Message passing layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Graph-level pooling
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


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load datasets
    train_dataset = GraphTokenDataset(
        root=args.graph_token_root,
        task=args.task,
        algorithm=args.algorithm,
        split='train',
        use_split_tasks_dirs=True,
    )
    val_dataset = GraphTokenDataset(
        root=args.graph_token_root,
        task=args.task,
        algorithm=args.algorithm,
        split='val',
        use_split_tasks_dirs=True,
    )
    test_dataset = GraphTokenDataset(
        root=args.graph_token_root,
        task=args.task,
        algorithm=args.algorithm,
        split='test',
        use_split_tasks_dirs=True,
    )

    print(f"#train: {len(train_dataset)} | #val: {len(val_dataset)} | #test: {len(test_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training examples found. Did you run the task generator?")
    if len(test_dataset) == 0:
        print(f"[warn] No test files found. Test metrics will be trivial.")

    # Show sample graph
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"[train] sample: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges, label={sample.y.item()}")

    # Create data loaders
    train_dl = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPNN(
        in_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.drop,
        pooling=args.pooling,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.watch(model, log="all", log_freq=100)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, crit, device)
        va_loss, va_acc = eval_epoch(model, val_dl, crit, device)
        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss, "val/acc": va_acc,
            "lr": opt.param_groups[0]["lr"],
        })
        print(f"epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"state_dict": best_state, "config": vars(args)},
                os.path.join(args.out_dir, f"best_{args.run_name}.pt"),
            )

    if best_state:
        model.load_state_dict(best_state)
    te_loss, te_acc = eval_epoch(model.to(device), test_dl, crit, device)
    print(f"TEST loss/acc: {te_loss:.4f}/{te_acc:.4f}")
    wandb.log({"test/loss": te_loss, "test/acc": te_acc})
    wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_token_root", type=str, default="graph-token", help="path to the graph-token repo")
    ap.add_argument("--task", type=str, default="cycle_check",
                    choices=["cycle_check","edge_existence","node_degree","node_count","edge_count","connected_nodes"])
    ap.add_argument("--algorithm", type=str, default="er", help="er/ba/sbm/… per the repo")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean","add","max"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs_mpnn")
    ap.add_argument("--wandb_project", type=str, default="graph-token")
    ap.add_argument("--run_name", type=str, default="mpnn-64x3-er")
    args = ap.parse_args()
    main(args)
