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

def parse_distance_label_from_text(text: str) -> Optional[int]:
    """Parse distance label from text like '<p> len3' -> 2 (0-indexed)

    Returns:
        Distance as 0-indexed class (len1->0, len2->1, len3->2, etc.),
        or None if unreachable (INF) or parsing fails
    """
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok == '<p>' and i + 1 < len(tokens):
            label_tok = tokens[i + 1].upper()  # Handle case-insensitive
            if label_tok in ('INF', 'INFINITY', '<EOS>'):
                # Unreachable - skip these samples for now
                return None
            if label_tok.startswith('LEN'):
                try:
                    distance = int(label_tok[3:])  # 'len3' -> 3
                    # Convert to 0-indexed for PyTorch (len1->0, len2->1, etc.)
                    return distance - 1
                except ValueError:
                    pass
    return None

def parse_query_nodes_from_text(text: str) -> Optional[Tuple[int, int]]:
    """Parse query nodes from text like '<q> shortest_distance 0 1' -> (0, 1)"""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok == '<q>' and i + 3 < len(tokens):
            # Expect format: <q> shortest_distance u v
            if tokens[i + 1] == 'shortest_distance':
                try:
                    u = int(tokens[i + 2])
                    v = int(tokens[i + 3])
                    return (u, v)
                except ValueError:
                    pass
    return None

def _extract_text_and_label(rec: Any, task: str = 'cycle_check') -> Tuple[Optional[str], Optional[int], Optional[Tuple[int, int]]]:
    """Extract text, label, and query nodes (for shortest_path) from a record.

    Returns:
        (text, label, query_nodes) where query_nodes is None for cycle_check
    """
    query_nodes = None

    if isinstance(rec, str):
        t = rec.strip()
        if task == 'shortest_path':
            lab = parse_distance_label_from_text(t)
            query_nodes = parse_query_nodes_from_text(t)
        else:
            lab = parse_yes_no_from_text(t)
        return t, lab, query_nodes

    if isinstance(rec, dict):
        text = rec.get("text") or rec.get("sequence")
        if text is None and "tokens" in rec and isinstance(rec["tokens"], (list, tuple)):
            text = " ".join(map(str, rec["tokens"]))
        lab = rec.get("label")

        # Parse label based on task
        if task == 'shortest_path':
            if isinstance(lab, int):
                pass  # Already an int distance
            elif isinstance(text, str):
                lab = parse_distance_label_from_text(text)
                query_nodes = parse_query_nodes_from_text(text)
        else:  # cycle_check or similar binary tasks
            if isinstance(lab, str):
                lab_l = lab.lower().strip()
                if lab_l in ("yes","true","connected","reachable"): lab = 1
                elif lab_l in ("no","false","disconnected","unreachable"): lab = 0
                else: lab = None
            elif isinstance(lab, (int, bool)):
                lab = int(bool(lab))
            if isinstance(text, str) and lab is None:
                lab = parse_yes_no_from_text(text)

        return (text.strip() if isinstance(text, str) else None), lab, query_nodes

    if isinstance(rec, list):
        if all(isinstance(x, (str, int)) for x in rec):
            t = " ".join(map(str, rec))
            if task == 'shortest_path':
                lab = parse_distance_label_from_text(t)
                query_nodes = parse_query_nodes_from_text(t)
            else:
                lab = parse_yes_no_from_text(t)
            return t, lab, query_nodes
        return None, None, None
    return None, None, None

def load_examples(path_glob: str, task: str = 'cycle_check', data_fraction: float = 1.0, seed: int = 0,
                  num_graphs: Optional[int] = None, num_pairs_per_graph: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load examples from JSON files.

    Args:
        path_glob: Glob pattern for files
        task: Task name ('cycle_check' or 'shortest_path')
        data_fraction: DEPRECATED - Fraction of data to use (0.0-1.0), use num_graphs instead
        seed: Random seed for reproducible sampling
        num_graphs: Number of graph files to sample (overrides data_fraction if set)
        num_pairs_per_graph: For shortest_path task, number of query pairs to sample per graph

    Returns:
        List of dicts with keys: text, label, and optionally query_u, query_v
    """
    import random
    files = sorted(glob(path_glob))

    # Sample graph files if num_graphs is specified
    if num_graphs is not None and len(files) > num_graphs:
        rng = random.Random(seed)
        files = rng.sample(files, num_graphs)
        files = sorted(files)  # Sort for reproducibility
        print(f"[load_examples] Sampled {num_graphs}/{len(sorted(glob(path_glob)))} graph files")

    out: List[Dict[str, Any]] = []

    def handle_obj(obj: Any):
        if isinstance(obj, list):
            for rec in obj:
                t, y, query_nodes = _extract_text_and_label(rec, task=task)
                if t:
                    entry = {"text": t, "label": y}
                    if query_nodes is not None:
                        entry["query_u"], entry["query_v"] = query_nodes
                    out.append(entry)
        else:
            t, y, query_nodes = _extract_text_and_label(obj, task=task)
            if t:
                entry = {"text": t, "label": y}
                if query_nodes is not None:
                    entry["query_u"], entry["query_v"] = query_nodes
                out.append(entry)

    # For shortest_path with num_pairs_per_graph, sample pairs per file
    if task == 'shortest_path' and num_pairs_per_graph is not None:
        rng = random.Random(seed)
        for fp in files:
            with open(fp, "r") as f:
                raw = f.read().strip()
            if not raw:
                continue

            file_examples = []
            def collect_obj(obj: Any):
                if isinstance(obj, list):
                    for rec in obj:
                        t, y, query_nodes = _extract_text_and_label(rec, task=task)
                        if t and query_nodes is not None:
                            entry = {"text": t, "label": y, "query_u": query_nodes[0], "query_v": query_nodes[1]}
                            file_examples.append(entry)
                else:
                    t, y, query_nodes = _extract_text_and_label(obj, task=task)
                    if t and query_nodes is not None:
                        entry = {"text": t, "label": y, "query_u": query_nodes[0], "query_v": query_nodes[1]}
                        file_examples.append(entry)

            try:
                obj = json.loads(raw)
                collect_obj(obj)
            except json.JSONDecodeError:
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        collect_obj(obj)
                    except json.JSONDecodeError:
                        t = line
                        lab = parse_distance_label_from_text(t)
                        query_nodes = parse_query_nodes_from_text(t)
                        if query_nodes:
                            entry = {"text": t, "label": lab, "query_u": query_nodes[0], "query_v": query_nodes[1]}
                            file_examples.append(entry)

            # Sample pairs from this graph file
            if len(file_examples) > num_pairs_per_graph:
                sampled = rng.sample(file_examples, num_pairs_per_graph)
                out.extend(sampled)
            else:
                out.extend(file_examples)

        print(f"[load_examples] Loaded {len(out)} pairs from {len(files)} graphs (target: {num_pairs_per_graph} pairs/graph)")
        return out

    # Original logic for other cases
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
                if task == 'shortest_path':
                    lab = parse_distance_label_from_text(t)
                    query_nodes = parse_query_nodes_from_text(t)
                    entry = {"text": t, "label": lab}
                    if query_nodes:
                        entry["query_u"], entry["query_v"] = query_nodes
                    out.append(entry)
                else:
                    out.append({"text": t, "label": parse_yes_no_from_text(t)})

    # Apply data fraction sampling if needed (deprecated)
    if num_graphs is None and data_fraction < 1.0 and len(out) > 0:
        rng = random.Random(seed)
        n_samples = max(1, int(len(out) * data_fraction))
        out = rng.sample(out, n_samples)

    return out


def balance_classes(examples: List[Dict[str, Any]], strategy: str = 'undersample', seed: int = 0) -> List[Dict[str, Any]]:
    """
    Balance class distribution using various strategies.

    Args:
        examples: List of example dicts with 'label' key
        strategy: Balancing strategy
            - 'undersample': Downsample all classes to match the minority class
            - 'median': Downsample to median class size (less aggressive)
            - 'oversample': Upsample minority classes to match the majority class (with replacement)
            - 'soft_oversample': Balance by downsampling majority and upsampling minority to mean
        seed: Random seed for reproducible sampling

    Returns:
        Balanced list of examples
    """
    from collections import defaultdict, Counter
    import random

    # Group examples by label
    label_to_examples = defaultdict(list)
    for ex in examples:
        label = ex.get('label')
        if label is not None:
            label_to_examples[label].append(ex)

    if not label_to_examples:
        return examples

    # Determine target size per class
    class_sizes = [len(exs) for exs in label_to_examples.values()]

    if strategy == 'undersample':
        # Match the minority class size
        target_size = min(class_sizes)
    elif strategy == 'median':
        # Use median size (less aggressive)
        import numpy as np
        target_size = int(np.median(class_sizes))
    elif strategy == 'oversample':
        # Match the majority class size
        target_size = max(class_sizes)
    elif strategy == 'soft_oversample':
        # Use mean as middle ground - downsample majority, upsample minority
        import numpy as np
        target_size = int(np.mean(class_sizes))
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}. Choose from: undersample, median, oversample, soft_oversample")

    # Balance each class
    rng = random.Random(seed)
    balanced = []

    print(f"\n⚖️  Class Balancing (strategy={strategy}):")
    print(f"  Target size per class: {target_size}")

    for label, exs in sorted(label_to_examples.items()):
        original_count = len(exs)

        if len(exs) == target_size:
            # Already at target size
            balanced.extend(exs)
            kept_count = len(exs)
        elif len(exs) > target_size:
            # Downsample (for undersample/median strategies)
            sampled = rng.sample(exs, target_size)
            balanced.extend(sampled)
            kept_count = target_size
        else:
            # Upsample (for oversample strategy or when class is smaller than target)
            # Duplicate examples with replacement to reach target size
            balanced.extend(exs)  # Add all original examples
            num_to_add = target_size - len(exs)
            duplicates = rng.choices(exs, k=num_to_add)  # Sample with replacement
            balanced.extend(duplicates)
            kept_count = target_size

        change_pct = 100 * kept_count / original_count if original_count > 0 else 100
        change_symbol = "↑" if kept_count > original_count else "↓" if kept_count < original_count else "="
        print(f"  Class {label}: {original_count:4d} → {kept_count:4d} ({change_symbol} {change_pct:.0f}%)")

    # Shuffle to mix classes
    rng.shuffle(balanced)

    print(f"  Total: {len(examples)} → {len(balanced)} examples")

    return balanced


def get_balanced_indices(dataset, strategy: str = 'undersample', seed: int = 0):
    """
    Get balanced indices for a PyG dataset by undersampling majority classes.

    Args:
        dataset: PyG dataset or list of Data objects
        strategy: Balancing strategy ('undersample' or 'median')
        seed: Random seed for reproducible sampling

    Returns:
        List of indices to keep for balanced dataset
    """
    from collections import defaultdict
    import random

    # Extract labels from dataset
    labels = []
    for i in range(len(dataset)):
        data = dataset[i]
        label = data.y.item() if hasattr(data, 'y') else None
        labels.append((i, label))

    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in labels:
        if label is not None:
            label_to_indices[label].append(idx)

    if not label_to_indices:
        return list(range(len(dataset)))

    # Determine target size per class
    class_sizes = [len(indices) for indices in label_to_indices.values()]

    if strategy == 'undersample':
        target_size = min(class_sizes)
    elif strategy == 'median':
        import numpy as np
        target_size = int(np.median(class_sizes))
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    # Subsample each class
    rng = random.Random(seed)
    balanced_indices = []

    print(f"\n⚖️  Class Balancing (strategy={strategy}):")
    print(f"  Target size per class: {target_size}")

    for label, indices in sorted(label_to_indices.items()):
        original_count = len(indices)
        if len(indices) <= target_size:
            # Keep all examples from minority classes
            balanced_indices.extend(indices)
            kept_count = len(indices)
        else:
            # Subsample majority classes
            sampled = rng.sample(indices, target_size)
            balanced_indices.extend(sampled)
            kept_count = target_size

        print(f"  Class {label}: {original_count} → {kept_count} ({100*kept_count/original_count:.0f}%)")

    # Shuffle to mix classes
    rng.shuffle(balanced_indices)

    print(f"  Total: {len(dataset)} → {len(balanced_indices)} examples")

    return balanced_indices


def load_examples_connected_nodes(path_glob: str, data_fraction: float = 1.0, seed: int = 0) -> List[Dict[str, Any]]:
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

    # Apply data fraction sampling if needed
    if data_fraction < 1.0 and len(out) > 0:
        import random
        rng = random.Random(seed)
        n_samples = max(1, int(len(out) * data_fraction))
        out = rng.sample(out, n_samples)

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
    """Resolve glob patterns for single algorithm (backward compatibility)."""
    train_base = os.path.join(root, "tasks_train", task, algorithm)
    test_base  = os.path.join(root, "tasks_test",  task, algorithm)
    train_glob_A = os.path.join(train_base, "train", "*.json")
    val_glob_A   = os.path.join(test_base, "val",   "*.json")  # Val should be in tasks_test
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
        # If no val directory, use test directory for validation
        val_glob = test_glob_A if use_split_tasks_dirs else test_glob_B
    return train_glob, val_glob, test_glob


def resolve_multi_algorithm_globs(root: str, task: str, train_algorithms: List[str],
                                    test_algorithm: str, use_split_tasks_dirs: bool = True):
    """Resolve glob patterns for multiple train algorithms and OOD test algorithm.

    Args:
        root: Root directory (graph-token repo path)
        task: Task name (cycle_check, shortest_path, etc.)
        train_algorithms: List of algorithms for training/validation (e.g., ['er', 'ba', 'sbm'])
        test_algorithm: Algorithm for OOD testing (e.g., 'ws')
        use_split_tasks_dirs: Use tasks_train/tasks_test structure

    Returns:
        train_globs: List of glob patterns for training data
        val_globs: List of glob patterns for validation data
        test_glob: Single glob pattern for test data (OOD algorithm)
    """
    def has_any(globpat): return len(glob(globpat)) > 0

    train_globs = []
    val_globs = []

    # Collect train/val globs from all train algorithms
    for algo in train_algorithms:
        train_base = os.path.join(root, "tasks_train", task, algo)
        test_base  = os.path.join(root, "tasks_test",  task, algo)
        train_glob_A = os.path.join(train_base, "train", "*.json")
        val_glob_A   = os.path.join(test_base, "val",   "*.json")
        base_B = os.path.join(root, "tasks", task, algo)
        train_glob_B = os.path.join(base_B, "train", "*.json")
        val_glob_B   = os.path.join(base_B, "val",   "*.json")

        if use_split_tasks_dirs and has_any(train_glob_A):
            train_globs.append(train_glob_A)
            val_glob = val_glob_A
        elif has_any(train_glob_B):
            train_globs.append(train_glob_B)
            val_glob = val_glob_B
        else:
            train_globs.append(train_glob_A)
            val_glob = val_glob_A

        if len(glob(val_glob)) == 0:
            # If no val directory, use test directory for validation
            test_glob_A = os.path.join(test_base, "test", "*.json")
            test_glob_B = os.path.join(base_B, "test", "*.json")
            val_glob = test_glob_A if use_split_tasks_dirs else test_glob_B

        val_globs.append(val_glob)

    # Get test glob from OOD algorithm
    test_base = os.path.join(root, "tasks_test", task, test_algorithm)
    test_glob_A = os.path.join(test_base, "test", "*.json")
    base_B = os.path.join(root, "tasks", task, test_algorithm)
    test_glob_B = os.path.join(base_B, "test", "*.json")

    if use_split_tasks_dirs and has_any(test_glob_A):
        test_glob = test_glob_A
    elif has_any(test_glob_B):
        test_glob = test_glob_B
    else:
        test_glob = test_glob_A

    return train_globs, val_globs, test_glob


def load_examples_multi_algorithm(root: str, task: str, algorithms: List[str], split: str,
                                    use_split_tasks_dirs: bool = True, seed: int = 0,
                                    num_graphs: Optional[int] = None, num_pairs_per_graph: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load examples from multiple graph generation algorithms.

    Args:
        root: Root directory (graph-token repo path)
        task: Task name
        algorithms: List of algorithms to load from
        split: Data split (train, val, test)
        use_split_tasks_dirs: Use tasks_train/tasks_test structure
        seed: Random seed
        num_graphs: Number of graphs to sample PER ALGORITHM
        num_pairs_per_graph: For shortest_path, number of pairs to sample per graph

    Returns:
        Combined list of examples from all algorithms
    """
    all_examples = []

    for algo in algorithms:
        _, _, _ = resolve_split_globs(root, task, algo, use_split_tasks_dirs)

        # Build glob pattern for this algorithm
        if use_split_tasks_dirs:
            if split in ['val', 'test']:
                base = os.path.join(root, 'tasks_test', task, algo)
            else:
                base = os.path.join(root, 'tasks_train', task, algo)
        else:
            base = os.path.join(root, 'tasks', task, algo)

        split_dir = os.path.join(base, split)
        if split == 'val' and len(glob(os.path.join(split_dir, '*.json'))) == 0:
            split_dir = os.path.join(base, 'test')

        path_glob = os.path.join(split_dir, '*.json')

        # Load examples from this algorithm
        examples = load_examples(path_glob, task=task, seed=seed + hash(algo) % 10000,
                                 num_graphs=num_graphs, num_pairs_per_graph=num_pairs_per_graph)
        print(f"  [{algo}] Loaded {len(examples)} examples")
        all_examples.extend(examples)

    print(f"[load_examples_multi_algorithm] Total: {len(all_examples)} examples from {len(algorithms)} algorithms")
    return all_examples


def determine_num_classes(examples: List[Dict[str, Any]], task: str) -> int:
    """Determine number of classes from the data.

    Args:
        examples: List of example dicts with 'label' key
        task: Task name ('cycle_check' or 'shortest_path')

    Returns:
        Number of classes
    """
    if task == 'cycle_check':
        return 2

    # For shortest_path, find max distance
    max_label = -1
    for ex in examples:
        label = ex.get('label')
        if label is not None and isinstance(label, int):
            max_label = max(max_label, label)

    # num_classes = max_label + 1 (since labels are 0-indexed)
    num_classes = max_label + 1

    # Convert 0-indexed back to actual distance for display
    max_distance = max_label + 1 if max_label >= 0 else 0

    print(f"\n{'='*80}")
    print(f"AUTO-DETERMINED NUM_CLASSES")
    print('='*80)
    print(f"Task: {task}")
    if task == 'shortest_path':
        print(f"Max path length found: len{max_distance} (0-indexed: {max_label})")
        print(f"Number of classes: {num_classes} (len1 to len{max_distance})")
    else:
        print(f"Number of classes: {num_classes}")
    print('='*80)
    print()

    return num_classes


def determine_num_classes_pyg(dataset, task: str) -> int:
    """Determine number of classes from PyG dataset.

    Args:
        dataset: PyG dataset or list of Data objects
        task: Task name ('cycle_check' or 'shortest_path')

    Returns:
        Number of classes
    """
    if task == 'cycle_check':
        return 2

    # For shortest_path, find max distance
    max_label = -1
    for i in range(len(dataset)):
        data = dataset[i]
        label = data.y.item() if hasattr(data, 'y') else None
        if label is not None:
            max_label = max(max_label, label)

    # num_classes = max_label + 1 (since labels are 0-indexed)
    num_classes = max_label + 1

    # Convert 0-indexed back to actual distance for display
    max_distance = max_label + 1 if max_label >= 0 else 0

    print(f"\n{'='*80}")
    print(f"AUTO-DETERMINED NUM_CLASSES")
    print('='*80)
    print(f"Task: {task}")
    if task == 'shortest_path':
        print(f"Max path length found: len{max_distance} (0-indexed: {max_label})")
        print(f"Number of classes: {num_classes} (len1 to len{max_distance})")
    else:
        print(f"Number of classes: {num_classes}")
    print('='*80)
    print()

    return num_classes