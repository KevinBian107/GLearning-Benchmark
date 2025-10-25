"""Loading the generated tokens from graph-token repo"""

import os, json
from glob import glob
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset

SPECIAL = ["<pad>", "<bos>", "<e>", "<n>", "<q>", "<p>", "<eos>", "yes", "no"]

def parse_yes_no_from_text(text: str) -> Optional[int]:
    for t in reversed(text.split()):
        t = t.lower()
        if t == "yes": return 1
        if t == "no": return 0
    return None

def _extract_text_and_label(rec: Any) -> Tuple[Optional[str], Optional[int]]:
    if isinstance(rec, str):
        t = rec.strip()
        return t, parse_yes_no_from_text(t)
    if isinstance(rec, dict):
        text = rec.get("text") or rec.get("sequence")
        if text is None and "tokens" in rec and isinstance(rec["tokens"], (list, tuple)):
            text = " ".join(map(str, rec["tokens"]))
        lab = rec.get("label")
        if isinstance(lab, str):
            lab_l = lab.lower().strip()
            if lab_l in ("yes","true","connected","reachable"): lab = 1
            elif lab_l in ("no","false","disconnected","unreachable"): lab = 0
            else: lab = None
        elif isinstance(lab, (int, bool)):
            lab = int(bool(lab))
        if isinstance(text, str) and lab is None:
            lab = parse_yes_no_from_text(text)
        return (text.strip() if isinstance(text, str) else None), lab
    if isinstance(rec, list):
        if all(isinstance(x, (str, int)) for x in rec):
            t = " ".join(map(str, rec))
            return t, parse_yes_no_from_text(t)
        return None, None
    return None, None

def load_examples(path_glob: str) -> List[Dict[str, Any]]:
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
        try:
            obj = json.loads(raw)
            handle_obj(obj)
            continue
        except json.JSONDecodeError:
            pass
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                handle_obj(obj)
            except json.JSONDecodeError:
                t = line
                out.append({"text": t, "label": parse_yes_no_from_text(t)})
    return out

def load_examples_connected_nodes(path_glob: str) -> List[Dict[str, Any]]:
    files = sorted(glob(path_glob))
    out: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp) as f:
            obj = json.load(f)
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if not isinstance(obj, dict):
            continue
        text = obj.get("text")
        if not isinstance(text, str):
            toks = obj.get("tokens")
            if isinstance(toks, list):
                text = " ".join(map(str, toks))
        if not text:
            continue
        u = obj.get("u", obj.get("src", obj.get("source", None)))
        v = obj.get("v", obj.get("dst", obj.get("target", None)))
        if (u is None or v is None) and "pair" in obj and isinstance(obj["pair"], (list, tuple)) and len(obj["pair"]) == 2:
            u, v = obj["pair"]
        lab = obj.get("label", obj.get("answer", obj.get("connected", None)))
        if isinstance(lab, str):
            lab_l = lab.lower().strip()
            if lab_l in ("yes","true","connected","reachable"): lab = 1
            elif lab_l in ("no","false","disconnected","unreachable"): lab = 0
            else: lab = None
        elif isinstance(lab, (int,bool)):
            lab = int(bool(lab))
        text_in = f"{text.strip()} <q> {u} {v} <p>" if u is not None and v is not None else text.strip()
        if lab is None:
            lab = parse_yes_no_from_text(text)
        out.append({"text": text_in, "label": lab, "u": u, "v": v})
    return out

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

class TokenDataset(Dataset):
    def __init__(self, examples, vocab, max_len=512, strip_label=True, require_label=True):
        self.vocab = vocab
        self.max_len = max_len
        self.seqs = []
        self.labels = []
        for ex in examples:
            text = ex["text"]
            label = ex.get("label")
            if label is None and isinstance(text, str):
                label = parse_yes_no_from_text(text)
            if require_label and label is None:
                continue
            toks = text.split()
            if strip_label and "<p>" in toks:
                p_idx = toks.index("<p>")
                toks = toks[:p_idx + 1]
            ids = [vocab.get(tok, vocab["<pad>"]) for tok in toks][:max_len]
            self.seqs.append(torch.tensor(ids, dtype=torch.long))
            self.labels.append(torch.tensor(int(label) if label is not None else 0, dtype=torch.long))
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx], self.labels[idx]

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

def resolve_split_globs(root: str, task: str, algorithm: str, use_split_tasks_dirs: bool = True):
    train_base = os.path.join(root, "tasks_train", task, algorithm)
    test_base  = os.path.join(root, "tasks_test",  task, algorithm)
    train_glob_A = os.path.join(train_base, "train", "*.json")
    val_glob_A   = os.path.join(train_base, "val",   "*.json")
    test_glob_A  = os.path.join(test_base,  "test",  "*.json")
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
        train_glob, val_glob, test_glob = train_glob_A, val_glob_A, test_glob_A
    if len(glob(val_glob)) == 0:
        val_glob = train_glob
    return train_glob, val_glob, test_glob