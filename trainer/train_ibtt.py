"""Train mini-transformer on tokenized graph"""

import os, argparse, random
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from graph_data_loader import (
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
        num_classes: int = 2,
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
        self.cls = nn.Linear(d_model, num_classes)

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

    # dataset globs (supports tasks/ or tasks_{train,test}/)
    train_glob, val_glob, test_glob = resolve_split_globs(
        root=dataset_cfg['graph_token_root'],
        task=dataset_cfg['task'],
        algorithm=dataset_cfg['algorithm'],
        use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
    )

    data_fraction = dataset_cfg.get('data_fraction', 1.0)
    seed = train_cfg['seed']

    train_ex = load_examples(train_glob, task=dataset_cfg['task'], data_fraction=data_fraction, seed=seed)
    val_ex = load_examples(val_glob, task=dataset_cfg['task'], data_fraction=data_fraction, seed=seed)
    test_ex = load_examples(test_glob, task=dataset_cfg['task'], data_fraction=data_fraction, seed=seed)
    
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

    # Determine number of classes based on task
    if dataset_cfg['task'] == 'shortest_path':
        num_classes = model_cfg.get('num_classes', 7)  # Default 7 for len1-len7
    else:  # cycle_check or other binary tasks
        num_classes = model_cfg.get('num_classes', 2)

    model = SimpleTransformer(
        vocab_size=len(vocab),
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        nlayers=model_cfg['nlayers'],
        d_ff=model_cfg['d_ff'],
        p_drop=model_cfg['dropout'],
        max_pos=model_cfg['max_pos'],
        num_classes=num_classes,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    crit = nn.CrossEntropyLoss()

    if wandb_cfg['use']:
        wandb.init(project=wandb_cfg['project'], name=output_cfg['run_name'], config=config)
        wandb.watch(model, log="all", log_freq=100)

    os.makedirs(output_cfg['out_dir'], exist_ok=True)
    best_val, best_state = -1.0, None

    for epoch in range(1, train_cfg['epochs'] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, crit, device)
        va_loss, va_acc = eval_epoch(model, val_dl,   crit, device)

        log_dict = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss,   "val/acc": va_acc,
            "lr": opt.param_groups[0]["lr"],
        }
        if wandb_cfg['use']:
            wandb.log(log_dict)

        print(f"epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"state_dict": best_state, "vocab": vocab, "config": config},
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
    ap = argparse.ArgumentParser(description="Train Graph Tokenization + Transformer on graph-token tasks")
    ap.add_argument("--config", type=str, default="configs/gtt_graph_token.yaml",
                    help="Path to YAML config file")
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Task: {config['dataset']['task']} | Algorithm: {config['dataset']['algorithm']}")

    main(config)