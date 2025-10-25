"""Train a mini-transformer upon the tokenized graph"""

import os, argparse, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from data_loader import (
    SPECIAL,
    load_examples,
    build_vocab_from_texts,
    TokenDataset,
    collate,
    resolve_split_globs,
)


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
    ):
        super().__init__()
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
        self.cls = nn.Linear(d_model, 2)

        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos.weight, std=0.02)
        nn.init.trunc_normal_(self.cls.weight, std=0.02)
        nn.init.zeros_(self.cls.bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = x.size()
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embed(x) + self.pos(pos_ids)
        key_pad = ~attn_mask  # True where to mask
        h = self.enc(h, src_key_padding_mask=key_pad)
        BOS_ID = SPECIAL.index("<bos>")
        if (x[:, 0] == BOS_ID).all():
            pooled = h[:, 0]
        else:
            lens = attn_mask.sum(-1, keepdim=True).clamp(min=1)
            pooled = (h * attn_mask.unsqueeze(-1)).sum(1) / lens
        z = self.norm(pooled)
        return self.cls(z)

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()

def train_one_epoch(model, dl, opt, crit, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, A, Y in dl:
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(X, A)
        loss = crit(logits, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = X.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, Y) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)

@torch.no_grad()
def eval_epoch(model, dl, crit, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, A, Y in dl:
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        logits = model(X, A)
        loss = crit(logits, Y)
        bs = X.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, Y) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve dataset globs (supports tasks/ or tasks_{train,test}/)
    train_glob, val_glob, test_glob = resolve_split_globs(
        root=args.graph_token_root,
        task=args.task,
        algorithm=args.algorithm,
        use_split_tasks_dirs=True,
    )

    train_ex = load_examples(train_glob)
    val_ex = load_examples(val_glob)
    test_ex = load_examples(test_glob)
    
    def show_sample(split_name, examples):
        if not examples:
            print(f"[{split_name}] no examples loaded")
            return
        ex = random.choice(examples)
        toks = ex["text"].split()
        print(f"[{split_name}] sample len={len(toks)}, label={ex.get('label','?')}")
        print(" ".join(toks[:40]) + (" ..." if len(toks) > 40 else ""))

    show_sample("train", train_ex)
    show_sample("val", val_ex)
    show_sample("test", test_ex)

    print(f"#train: {len(train_ex)} | #val: {len(val_ex)} | #test: {len(test_ex)}")
    if len(train_ex) == 0:
        raise RuntimeError(f"No training examples found at {train_glob}. Did you run the task generator?")
    if len(test_ex) == 0:
        print(f"[warn] No test files at {test_glob}. Test metrics will be trivial.")

    vocab, _ = build_vocab_from_texts([e["text"] for e in train_ex], max_tokens=args.max_vocab)
    pad_id = vocab["<pad>"]

    train_ds = TokenDataset(train_ex, vocab, args.max_len)
    val_ds = TokenDataset(val_ex,   vocab, args.max_len)
    test_ds = TokenDataset(test_ex,  vocab, args.max_len)

    coll = lambda b: collate(b, pad_id)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=2, collate_fn=coll)
    val_dl = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=coll)
    test_dl = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=coll)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        d_ff=args.d_ff,
        p_drop=args.drop,
        max_pos=args.max_len,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.watch(model, log="all", log_freq=100)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, crit, device)
        va_loss, va_acc = eval_epoch(model, val_dl,   crit, device)
        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss,   "val/acc": va_acc,
            "lr": opt.param_groups[0]["lr"],
        })
        print(f"epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"state_dict": best_state, "vocab": vocab},
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
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_vocab", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs_cyclecheck")
    ap.add_argument("--wandb_project", type=str, default="graph-token-cyclecheck")
    ap.add_argument("--run_name", type=str, default="tt-256x4-er")
    args = ap.parse_args()
    main(args)