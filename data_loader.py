"""Loading the generated tokens from graph-token repo"""

import os, json
from glob import glob
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset

SPECIAL = ["<pad>", "<bos>", "<e>", "<n>", "<q>", "<p>", "<eos>", "yes", "no"]

def _extract_text_and_label(rec: Any) -> Tuple[Optional[str], Optional[int]]:
    """Return (text, label) from a dict/list/str record; label is 1 for 'yes', 0 for 'no', or None."""
    if isinstance(rec, str):
        return rec.strip(), None

    if isinstance(rec, dict):
        text = rec.get("text") or rec.get("sequence")
        if text is None and "tokens" in rec and isinstance(rec["tokens"], (list, tuple)):
            text = " ".join(map(str, rec["tokens"]))

        lab = rec.get("label")
        if isinstance(lab, str):
            lab_l = lab.lower().strip()
            lab = 1 if lab_l == "yes" else 0 if lab_l == "no" else None
        elif isinstance(lab, (int, bool)):
            lab = int(bool(lab))
        else:
            lab = None

        return (text.strip() if isinstance(text, str) else None), lab

    if isinstance(rec, list):
        if all(isinstance(x, (str, int)) for x in rec):
            return " ".join(map(str, rec)), None
        return None, None

    return None, None


def load_examples(path_glob: str) -> List[Dict[str, Any]]:
    """
    Returns [{"text": str, "label": Optional[int]}, ...]
    Supports per-file JSON object/array, or JSONL (one object per line).
    """
    files = sorted(glob(path_glob))
    out: List[Dict[str, Any]] = []

    def handle_obj(obj: Any):
        if isinstance(obj, list):
            for rec in obj:
                t, y = _extract_text_and_label(rec)
                if t: out.append({"text": t, "label": y})
        else:
            t, y = _extract_text_and_label(obj)
            if t: out.append({"text": t, "label": y})

    for fp in files:
        with open(fp, "r") as f:
            raw = f.read().strip()
        if not raw:
            continue

        # try full JSON
        try:
            obj = json.loads(raw)
            handle_obj(obj)
            continue
        except json.JSONDecodeError:
            pass

        # fallback: JSONL
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                handle_obj(obj)
            except json.JSONDecodeError:
                out.append({"text": line, "label": None})

    return out


# Vocab & labels
def build_vocab_from_texts(texts: List[str], min_freq: int = 1, max_tokens: Optional[int] = None):
    cnt = Counter()
    for text in texts:
        cnt.update(text.split())

    vocab = {tok: i for i, tok in enumerate(SPECIAL)}
    idx = len(vocab)
    for tok, c in cnt.most_common():
        if tok in vocab: continue
        if c < min_freq: break
        vocab[tok] = idx; idx += 1
        if max_tokens and idx >= max_tokens: break
    itos = {i: t for t, i in vocab.items()}
    return vocab, itos


def get_label_from_text_or_fallback(text: str, explicit_label: Optional[int]) -> int:
    if explicit_label is not None:
        return int(bool(explicit_label))
    toks = text.split()
    try:
        p = len(toks) - 1 - toks[::-1].index("<p>")
        lab = toks[p + 1].lower()
        return 1 if lab == "yes" else 0
    except Exception:
        for t in reversed(toks):
            if t.lower() in ("yes", "no"):
                return 1 if t.lower() == "yes" else 0
    return 0


# Dataset & Collate
class TokenDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], vocab: Dict[str, int], max_len: int = 512):
        self.vocab = vocab
        self.max_len = max_len
        self.seqs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        for ex in examples:
            text = ex["text"]
            label = get_label_from_text_or_fallback(text, ex.get("label"))
            ids = [vocab.get(tok, vocab["<pad>"]) for tok in text.split()][:max_len]
            self.seqs.append(torch.tensor(ids, dtype=torch.long))
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int):
        return self.seqs[i], self.labels[i]


def collate(batch, pad_id: int):
    xs, ys = zip(*batch)
    L = max(x.size(0) for x in xs)
    X = torch.full((len(xs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(xs), L), dtype=torch.bool)
    for i, x in enumerate(xs):
        X[i, :x.size(0)] = x
        attn[i, :x.size(0)] = True
    Y = torch.tensor(ys, dtype=torch.long)
    return X, attn, Y


def resolve_split_globs(
    root: str,
    task: str,
    algorithm: str,
    use_split_tasks_dirs: bool = True,
):
    """
    Returns (train_glob, val_glob, test_glob) supporting:
      - graph-token/tasks/<task>/<algorithm>/{train,val,test}/*.json
      - graph-token/tasks_train/<task>/<algorithm>/train/*.json and
        graph-token/tasks_test/<task>/<algorithm>/test/*.json
    If val not present, falls back to train as validation.
    """
    # Option A: split trees
    train_base = os.path.join(root, "tasks_train", task, algorithm)
    test_base  = os.path.join(root, "tasks_test",  task, algorithm)
    train_glob_A = os.path.join(train_base, "train", "*.json")
    val_glob_A   = os.path.join(train_base, "val",   "*.json")
    test_glob_A  = os.path.join(test_base,  "test",  "*.json")

    # Option B: single tree
    base_B = os.path.join(root, "tasks", task, algorithm)
    train_glob_B = os.path.join(base_B, "train", "*.json")
    val_glob_B   = os.path.join(base_B, "val",   "*.json")
    test_glob_B  = os.path.join(base_B, "test",  "*.json")

    def has_any(globpat): return len(glob(globpat)) > 0

    if use_split_tasks_dirs and has_any(train_glob_A):
        train_glob, val_glob, test_glob = train_glob_A, val_glob_A, test_glob_A
    elif has_any(train_glob_B):
        train_glob, val_glob, test_glob = train_glob_B, val_glob_B, test_glob_B
    else:
        # fallback to guess split trees anyway
        train_glob, val_glob, test_glob = train_glob_A, val_glob_A, test_glob_A

    # If val is empty, use train as val (you can change to a held-out slice if you prefer)
    if len(glob(val_glob)) == 0:
        val_glob = train_glob
    return train_glob, val_glob, test_glob
