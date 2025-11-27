"""
Train Transformer on graph-token tasks using AutoGraph's tokenization.
"""

import os
import sys
import argparse
import random
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

# Since train_agtt.py is in trainer/, need to go up one level to find AutoGraph/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
autograph_root = os.path.join(parent_dir, 'AutoGraph')
sys.path.insert(0, autograph_root)

from autograph.datamodules.data.tokenizer import Graph2TrailTokenizer
from graph_data_loader import (
    GraphTokenDatasetForAutoGraph,
    determine_num_classes_pyg,
    ZINCDatasetForAutoGraph,
    get_zinc_num_types,
    build_fixed_zinc_vocab,
    get_atom_type_id,
    get_bond_type_id,
)
from torch.utils.data import Subset
from trainer.metrics import compute_metrics, aggregate_metrics, format_confusion_matrix, get_loss_function, create_confusion_matrix_heatmap


class SimpleTransformer(nn.Module):
    """Same transformer architecture as train_ibtt.py for fair comparison."""
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
        tokenizer_idx_offset: int = 4,
        task: str = 'cycle_check',
    ):
        super().__init__()
        self.d_model = d_model
        self.use_query_nodes = use_query_nodes
        self.tokenizer_idx_offset = tokenizer_idx_offset
        self.task = task

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

    def extract_query_nodes(self, x: torch.Tensor, h: torch.Tensor, q_token_id: int) -> tuple:
        """
        Extract query node embeddings by finding <q> marker token.

        Query tokens are appended as: [...trail...] <q> u v
        This method searches for <q> and extracts u and v at fixed offsets.

        Args:
            x: Token IDs (batch, seq_len)
            h: Transformer hidden states (batch, seq_len, d_model)
            q_token_id: Token ID for <q> marker

        Returns:
            (u_emb, v_emb): Query node embeddings (batch, d_model) each
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize with zeros (fallback if query tokens not found)
        u_emb = torch.zeros(batch_size, self.d_model, device=device)
        v_emb = torch.zeros(batch_size, self.d_model, device=device)

        # Find <q> marker and extract u, v at offsets +1, +2
        for b in range(batch_size):
            # Find <q> token position
            q_positions = (x[b] == q_token_id).nonzero(as_tuple=True)[0]

            if len(q_positions) > 0:
                q_pos = q_positions[0].item()

                # Extract u and v at fixed offsets from <q>
                # Format: ... <q> u v
                if q_pos + 2 < x.size(1):
                    u_emb[b] = h[b, q_pos + 1]  # Next position after <q>
                    v_emb[b] = h[b, q_pos + 2]  # Two positions after <q>

        return u_emb, v_emb

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, data_list=None) -> torch.Tensor:
        B, L = x.size()
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embed(x) + self.pos(pos_ids)
        key_pad = ~attn_mask  # True where to mask
        h = self.enc(h, src_key_padding_mask=key_pad)

        # Get SOS embedding
        sos_emb = h[:, 0]

        # Extract query node embeddings if enabled
        if self.use_query_nodes and data_list is not None:
            # Compute <q> token ID from the first data in batch
            # q_token_id = tokenizer.idx_offset + max_num_nodes
            # Since all graphs in batch should have same or less nodes, use first data's num_nodes
            q_token_id = self.tokenizer_idx_offset + data_list[0].num_nodes

            u_emb, v_emb = self.extract_query_nodes(x, h, q_token_id)
            # Apply LayerNorm to each embedding separately, then concatenate
            sos_normed = self.norm(sos_emb)
            u_normed = self.norm(u_emb)
            v_normed = self.norm(v_emb)
            pooled = torch.cat([sos_normed, u_normed, v_normed], dim=-1)
        else:
            pooled = self.norm(sos_emb)

        out = self.cls(pooled)

        # For regression tasks, squeeze the output
        if self.task == 'zinc':
            return out.squeeze(-1)
        return out


class TokenizedGraphDataset(Dataset):
    """
    Wraps PyG dataset and tokenizes graphs on-the-fly using AutoGraph tokenizer.

    NEW: For ZINC tasks, remaps AutoGraph token IDs to fixed vocabulary IDs
    to ensure consistent feature encoding with IBTT.
    """
    def __init__(self, pyg_dataset, tokenizer, task='cycle_check', remap_to_fixed_vocab=False):
        self.pyg_dataset = pyg_dataset
        self.tokenizer = tokenizer
        self.task = task
        self.remap_to_fixed_vocab = remap_to_fixed_vocab

        # Build fixed vocabulary mapping if remapping is enabled
        if remap_to_fixed_vocab:
            self.fixed_vocab, _ = build_fixed_zinc_vocab()
            print("[TokenizedGraphDataset] Using fixed vocabulary remapping for ZINC")

    def __len__(self):
        return len(self.pyg_dataset)

    def remap_zinc_tokens(self, tokens, data):
        """
        Remap AutoGraph's token IDs to fixed vocabulary IDs for ZINC.

        AutoGraph structure for labeled graphs:
        - Special tokens: 0-5 (SOS, RESET, LADJ, RADJ, EOS, PAD)
        - Node positions: idx_offset to idx_offset + max_nodes - 1
        - Atom types: node_idx_offset to node_idx_offset + num_atom_types - 1
        - Bond types: edge_idx_offset to edge_idx_offset + num_edge_types - 1

        We remap atom/bond types to fixed vocabulary IDs:
        - Atom type 0 (C) → ID 8
        - Atom type 1 (N) → ID 9
        - etc.
        """
        remapped = []

        # Get AutoGraph's offsets
        node_idx_offset = self.tokenizer.node_idx_offset
        edge_idx_offset = self.tokenizer.edge_idx_offset
        idx_offset = self.tokenizer.idx_offset

        for token_id in tokens.tolist():
            # Special tokens mapping
            if token_id == 0:  # SOS
                remapped.append(self.fixed_vocab['<bos>'])
            elif token_id == 1:  # RESET (special AutoGraph token)
                remapped.append(self.fixed_vocab['<pad>'])  # Map to pad
            elif token_id == 2:  # LADJ (left adjacency)
                remapped.append(self.fixed_vocab['<pad>'])  # Map to pad
            elif token_id == 3:  # RADJ (right adjacency)
                remapped.append(self.fixed_vocab['<pad>'])  # Map to pad
            elif token_id == 4:  # EOS
                remapped.append(self.fixed_vocab['<eos>'])
            elif token_id == 5:  # PAD
                remapped.append(self.fixed_vocab['<pad>'])

            # Atom types (node labels)
            elif node_idx_offset <= token_id < edge_idx_offset:
                atom_type_idx = token_id - node_idx_offset
                # Map to fixed vocab ID (8 for C, 9 for N, etc.)
                try:
                    fixed_id = get_atom_type_id(atom_type_idx)
                    remapped.append(fixed_id)
                except ValueError:
                    # Out of range atom type, use a safe fallback
                    remapped.append(22 + token_id)

            # Bond types (edge labels)
            elif token_id >= edge_idx_offset:
                bond_type_idx = token_id - edge_idx_offset + 1  # +1 because bonds are 1-indexed
                # Map to fixed vocab ID (17 for single, 18 for double, etc.)
                # Handle out-of-range gracefully (AutoGraph may generate extra tokens)
                try:
                    fixed_id = get_bond_type_id(bond_type_idx)
                    remapped.append(fixed_id)
                except ValueError:
                    # Out of range bond type (e.g., special tokens)
                    # Map to a safe range beyond fixed vocab
                    remapped.append(22 + token_id)

            # Node positions (keep as-is, they don't affect chemistry)
            # These are between idx_offset and node_idx_offset
            else:
                # For node positions, offset them to avoid collision with fixed vocab
                # Fixed vocab uses 0-21, so start node positions at 22+
                if idx_offset <= token_id < node_idx_offset:
                    # Remap to start at 22
                    remapped.append(22 + (token_id - idx_offset))
                else:
                    # Fallback: use safe offset
                    remapped.append(22 + token_id)

        return torch.tensor(remapped, dtype=torch.long)

    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]
        # Tokenize the graph using AutoGraph's SENT tokenizer
        # The tokenizer returns a 1D tensor of token IDs
        tokens = self.tokenizer(data)

        # Remap to fixed vocabulary for ZINC tasks
        if self.remap_to_fixed_vocab:
            tokens = self.remap_zinc_tokens(tokens, data)

        # For shortest_path, append query tokens: <q> shortest_distance u v
        if self.task == 'shortest_path' and hasattr(data, 'query_u') and hasattr(data, 'query_v'):
            # Special token IDs (after graph tokens)
            # AutoGraph uses: 0=PAD, 1=SOS, 2=EOS, 3=MASK, 4+=node IDs
            # We'll append query tokens with special IDs
            query_token_id = self.tokenizer.idx_offset + data.num_nodes  # <q>
            u_token_id = self.tokenizer.idx_offset + data.query_u  # node u
            v_token_id = self.tokenizer.idx_offset + data.query_v  # node v

            # Append query tokens: <q> u v
            query_tokens = torch.tensor([query_token_id, u_token_id, v_token_id], dtype=torch.long)
            tokens = torch.cat([tokens, query_tokens])

        # Create attention mask (all True for valid tokens)
        attn_mask = torch.ones(tokens.size(0), dtype=torch.bool)
        label = data.y.item()
        # Return the original PyG data object as well for query node extraction
        return tokens, attn_mask, label, data


def collate_fn(batch):
    """Collate function to batch tokenized graphs."""
    tokens_list, attn_list, labels, data_list = zip(*batch)

    # Find max length
    max_len = max(t.size(0) for t in tokens_list)
    batch_size = len(tokens_list)

    # Create padded tensors
    tokens_padded = torch.full((batch_size, max_len),
                               Graph2TrailTokenizer.pad, dtype=torch.long)
    attn_padded = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (tokens, attn) in enumerate(zip(tokens_list, attn_list)):
        seq_len = tokens.size(0)
        tokens_padded[i, :seq_len] = tokens
        attn_padded[i, :seq_len] = attn

    # Check if labels are floats (regression) or ints (classification)
    # ZINC has float labels, graph-token tasks have int labels
    if isinstance(labels[0], float):
        labels_tensor = torch.tensor(labels, dtype=torch.float)
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Return data_list for query node extraction
    return tokens_padded, attn_padded, labels_tensor, list(data_list)


def train_one_epoch(model, dl, opt, crit, device, task='cycle_check'):
    model.train()
    all_metrics = []
    for X, A, Y, data_list in dl:
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(X, A, data_list=data_list)
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
def eval_epoch(model, dl, crit, device, task='cycle_check'):
    model.eval()
    all_metrics = []
    for X, A, Y, data_list in dl:
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        logits = model(X, A, data_list=data_list)
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

    print("=" * 80)
    print("Training AGTT: AutoGraph Tokenization + Transformer")
    print("=" * 80)
    print(f"Task: {dataset_cfg['task']}")
    print()

    # Check if task is ZINC molecular property prediction
    is_zinc = dataset_cfg['task'] == 'zinc'

    if is_zinc:
        # Load ZINC dataset
        print(f"\n{'='*80}")
        print(f"LOADING DATA - ZINC Molecular Property Prediction")
        print('='*80)
        print(f"Task: Graph-level regression (constrained solubility)")
        print()

        train_pyg = ZINCDatasetForAutoGraph(
            zinc_root=dataset_cfg['zinc_root'],
            split='train',
            subset=dataset_cfg.get('subset', True),
        )

        val_pyg = ZINCDatasetForAutoGraph(
            zinc_root=dataset_cfg['zinc_root'],
            split='val',
            subset=dataset_cfg.get('subset', True),
        )

        test_pyg = ZINCDatasetForAutoGraph(
            zinc_root=dataset_cfg['zinc_root'],
            split='test',
            subset=dataset_cfg.get('subset', True),
        )

        print(f"#train: {len(train_pyg)} | #val: {len(val_pyg)} | #test: {len(test_pyg)}")

        # Log example molecule
        if len(train_pyg) > 0:
            sample = train_pyg[0]
            print(f"\n[EXAMPLE MOLECULE]")
            print(f"  Num atoms: {sample.num_nodes}")
            print(f"  Num bonds: {sample.edge_index.size(1)}")
            print(f"  Target (y): {sample.y.item():.4f}")
            print(f"  Atom types (first 10): {sample.x[:10].flatten().tolist()}")
            print(f"  Bond types (first 10): {sample.edge_attr[:10].tolist()}")
        print('='*80)
        print()

    else:
        # Load PyG datasets (graph-native format) from multiple algorithms
        print("Loading graph-native datasets...")
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

        train_pyg = GraphTokenDatasetForAutoGraph(
            root=dataset_cfg['graph_token_root'],
            task=dataset_cfg['task'],
            algorithm=train_algorithms,
            split='train',
            use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
            num_graphs=num_graphs,
            num_pairs_per_graph=num_pairs_per_graph,
            seed=seed,
        )

        val_pyg = GraphTokenDatasetForAutoGraph(
            root=dataset_cfg['graph_token_root'],
            task=dataset_cfg['task'],
            algorithm=train_algorithms,
            split='val',
            use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
            num_graphs=num_graphs,
            num_pairs_per_graph=num_pairs_per_graph,
            seed=seed,
        )

        test_pyg = GraphTokenDatasetForAutoGraph(
            root=dataset_cfg['graph_token_root'],
            task=dataset_cfg['task'],
            algorithm=[test_algorithm],
            split='test',
            use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
            num_graphs=num_graphs,
            num_pairs_per_graph=num_pairs_per_graph,
            seed=seed,
        )

        print(f"#train: {len(train_pyg)} | #val: {len(val_pyg)} | #test: {len(test_pyg)}")

        if len(train_pyg) == 0:
            raise RuntimeError(f"No training examples found. Did you run the task generator?")

        # Log example graphs from each algorithm
        print(f"\n{'='*80}")
        print(f"EXAMPLE GRAPHS FROM EACH ALGORITHM")
        print('='*80)

        from trainer.metrics import log_graph_examples
        for algo in train_algorithms:
            # Create a small dataset from this algorithm for display
            algo_dataset = GraphTokenDatasetForAutoGraph(
                root=dataset_cfg['graph_token_root'],
                task=dataset_cfg['task'],
                algorithm=[algo],
                split='train',
                use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
                num_graphs=1,
                num_pairs_per_graph=1,
                seed=seed,
            )
            if len(algo_dataset) > 0:
                print(f"\n[TRAIN - {algo.upper()}]")
                print(log_graph_examples(algo_dataset, task=dataset_cfg['task'], num_examples=1))

        # Show one example from test (OOD) algorithm
        test_display_dataset = GraphTokenDatasetForAutoGraph(
            root=dataset_cfg['graph_token_root'],
            task=dataset_cfg['task'],
            algorithm=[test_algorithm],
            split='test',
            use_split_tasks_dirs=dataset_cfg['use_split_tasks_dirs'],
            num_graphs=1,
            num_pairs_per_graph=1,
            seed=seed,
        )
        if len(test_display_dataset) > 0:
            print(f"\n[TEST - {test_algorithm.upper()} (OOD)]")
            print(log_graph_examples(test_display_dataset, task=dataset_cfg['task'], num_examples=1))

        print('='*80)
        print()

    # Initialize AutoGraph's tokenizer
    print("\nInitializing AutoGraph tokenizer...")

    # For ZINC, use labeled_graph=True to preserve atom and bond types
    # For synthetic graphs, use labeled_graph=False (structure only)
    if is_zinc:
        # ZINC molecules have labeled nodes (atom types) and edges (bond types)
        num_node_types, num_edge_types = get_zinc_num_types()
        print(f"ZINC atom types: {num_node_types}, bond types: {num_edge_types}")

        tokenizer = Graph2TrailTokenizer(
            dataset_names=[],
            max_length=dataset_cfg['max_len'],
            truncation_length=dataset_cfg['max_len'],
            labeled_graph=True,  # Preserve atom and bond types
            undirected=True,
        )
        print("Using chemistry-aware tokenization (labeled_graph=True)")
    else:
        # Synthetic graphs are unlabeled (structure only)
        tokenizer = Graph2TrailTokenizer(
            dataset_names=[],
            max_length=dataset_cfg['max_len'],
            truncation_length=dataset_cfg['max_len'],
            labeled_graph=False,  # Structure only
            undirected=True,
        )
        print("Using structure-only tokenization (labeled_graph=False)")

    # Set maximum number of nodes for tokenizer
    max_num_nodes = max(data.num_nodes for data in train_pyg)
    tokenizer.set_num_nodes(max_num_nodes)
    print(f"Max nodes: {max_num_nodes}")

    # For labeled graphs, set node and edge types (must be done AFTER set_num_nodes)
    if is_zinc:
        tokenizer.set_num_node_and_edge_types(num_node_types, num_edge_types)
        print(f"Node idx offset: {tokenizer.node_idx_offset}")
        print(f"Edge idx offset: {tokenizer.edge_idx_offset}")

    # Calculate vocabulary size
    # For ZINC: Use FIXED vocabulary size to match IBTT encoding
    # For other tasks: Use AutoGraph's dynamic vocabulary
    if is_zinc:
        print("\n" + "=" * 80)
        print("BUILDING FIXED ZINC VOCABULARY (TO MATCH IBTT)")
        print("=" * 80)
        print("Using fixed vocabulary for consistent encoding with IBTT")
        print("Chemistry features will use the same embedding indices across methods")
        print()

        # Build fixed vocabulary
        fixed_vocab, _ = build_fixed_zinc_vocab()

        # Estimate max vocab size needed (fixed + dynamic tokens)
        # Fixed: 22 tokens (special + atoms + bonds + task)
        # Dynamic: node positions (max_num_nodes)
        vocab_size = len(fixed_vocab) + max_num_nodes + 100  # +100 buffer for safety
        print(f"  Fixed tokens (special + chemistry): {len(fixed_vocab)}")
        print(f"  Max nodes (dynamic): {max_num_nodes}")
        print(f"  Total vocabulary size: {vocab_size}")
        print()

        # Verify chemistry features are at fixed positions
        print("Chemistry Feature Encoding (matching IBTT):")
        print(f"  Carbon (C):       ID {fixed_vocab['C']}")
        print(f"  Nitrogen (N):     ID {fixed_vocab['N']}")
        print(f"  Oxygen (O):       ID {fixed_vocab['O']}")
        print(f"  Single bond:      ID {fixed_vocab['single']}")
        print(f"  Double bond:      ID {fixed_vocab['double']}")
        print()

        print("AutoGraph's original offsets (before remapping):")
        print(f"  Node idx offset: {tokenizer.node_idx_offset}")
        print(f"  Edge idx offset: {tokenizer.edge_idx_offset}")
        print("=" * 80)
        print()

        # Enable remapping for ZINC datasets
        remap_to_fixed_vocab = True
    else:
        # For unlabeled graphs: special tokens + node IDs + query token (<q>)
        vocab_size = tokenizer.idx_offset + max_num_nodes + 1  # +1 for <q> token
        print(f"Vocabulary size: {vocab_size}")
        remap_to_fixed_vocab = False

    print()

    # Wrap datasets with tokenization
    # For ZINC, enable remapping to fixed vocabulary
    train_ds = TokenizedGraphDataset(train_pyg, tokenizer, task=dataset_cfg['task'], remap_to_fixed_vocab=remap_to_fixed_vocab)
    val_ds = TokenizedGraphDataset(val_pyg, tokenizer, task=dataset_cfg['task'], remap_to_fixed_vocab=remap_to_fixed_vocab)
    test_ds = TokenizedGraphDataset(test_pyg, tokenizer, task=dataset_cfg['task'], remap_to_fixed_vocab=remap_to_fixed_vocab)

    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=train_cfg['batch_size'],
                         shuffle=True, num_workers=train_cfg['num_workers'],
                         collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=train_cfg['batch_size'],
                       shuffle=False, num_workers=train_cfg['num_workers'],
                       collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=train_cfg['batch_size'],
                        shuffle=False, num_workers=train_cfg['num_workers'],
                        collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Auto-determine number of classes from ALL data (train, val, test combined)
    from torch.utils.data import ConcatDataset
    all_pyg = ConcatDataset([train_pyg, val_pyg, test_pyg])
    num_classes = determine_num_classes_pyg(all_pyg, task=dataset_cfg['task'])

    # Determine if we should use query node embeddings
    if dataset_cfg['task'] == 'shortest_path':
        use_query_nodes = True  # Enable query node embeddings (appended to trail)
    else:  # cycle_check or other binary tasks
        use_query_nodes = False  # Not needed for global properties

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        nlayers=model_cfg['nlayers'],
        d_ff=model_cfg['d_ff'],
        p_drop=model_cfg['dropout'],
        max_pos=model_cfg['max_pos'],
        num_classes=num_classes,
        use_query_nodes=use_query_nodes,
        tokenizer_idx_offset=tokenizer.idx_offset,
        task=dataset_cfg['task'],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    if use_query_nodes:
        print(f"Using query node embeddings (classifier input: {3 * model_cfg['d_model']})")
        print("Query extraction: Search for <q> marker, extract u and v at offsets +1, +2")
    else:
        print(f"Using single embedding (classifier input: {model_cfg['d_model']})")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'],
                           weight_decay=train_cfg['weight_decay'])

    # Select loss function based on task
    crit = get_loss_function(dataset_cfg['task'], device)

    # Create run name with training algorithms in parentheses (or task name for ZINC)
    if is_zinc:
        wandb_run_name = output_cfg['run_name']
    else:
        algo_str = '+'.join(train_algorithms)
        wandb_run_name = f"{output_cfg['run_name']} ({algo_str})"

    if wandb_cfg['use']:
        wandb.init(project=wandb_cfg['project'], name=wandb_run_name, config=config)
        wandb.watch(model, log="all", log_freq=100)
        wandb.log({"model/num_parameters": num_params})

    os.makedirs(output_cfg['out_dir'], exist_ok=True)
    best_val, best_state = -1.0, None

    # Timing and efficiency tracking
    training_start_time = time.time()
    num_train_graphs = len(train_ds)
    initial_val_acc = 0.0
    time_to_best = 0.0

    print("Starting training...")
    print("=" * 80)

    for epoch in range(1, train_cfg['epochs'] + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_dl, opt, crit, device, task=dataset_cfg['task'])
        val_metrics = eval_epoch(model, val_dl, crit, device, task=dataset_cfg['task'])

        epoch_duration = time.time() - epoch_start

        # Extract key metrics
        tr_loss = train_metrics['loss']
        va_loss = val_metrics['loss']

        # Calculate throughput (graphs per second)
        graphs_per_sec = num_train_graphs / epoch_duration if epoch_duration > 0 else 0

        # GPU memory tracking
        gpu_mem_allocated = 0
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        # Build logging dictionary (different for regression vs classification)
        if dataset_cfg['task'] == 'zinc':
            # For regression: track MAE as primary metric (lower is better)
            tr_mae = train_metrics.get('mae', 0)
            va_mae = val_metrics.get('mae', 0)

            # Track MAE improvement for efficiency
            if initial_val_acc == 0.0:
                initial_val_acc = va_mae
            mae_improvement = initial_val_acc - va_mae  # Positive if improving
            elapsed_time = time.time() - training_start_time
            time_per_improvement = elapsed_time / mae_improvement if mae_improvement > 0.01 else 0

            log_dict = {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/mae": tr_mae,
                "train/mse": train_metrics.get('mse', 0),
                "train/rmse": train_metrics.get('rmse', 0),
                "val/loss": va_loss,
                "val/mae": va_mae,
                "val/mse": val_metrics.get('mse', 0),
                "val/rmse": val_metrics.get('rmse', 0),
                "lr": opt.param_groups[0]["lr"],
                "time/epoch_duration": epoch_duration,
                "throughput/graphs_per_sec": graphs_per_sec,
                "memory/gpu_allocated_mb": gpu_mem_allocated,
                "efficiency/time_per_mae_improvement": time_per_improvement,
            }
        else:
            # For classification: track accuracy
            tr_acc = train_metrics['accuracy']
            va_acc = val_metrics['accuracy']

            # Track accuracy gain for efficiency
            acc_gain = (va_acc - initial_val_acc) * 100  # Convert to percentage
            elapsed_time = time.time() - training_start_time
            time_per_1pct_acc = elapsed_time / acc_gain if acc_gain > 0 else 0

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

            # Add MSE/MAE for shortest_path (also regression-like)
            if dataset_cfg['task'] == 'shortest_path':
                log_dict["train/mse"] = train_metrics.get('mse', 0)
                log_dict["train/mae"] = train_metrics.get('mae', 0)
                log_dict["val/mse"] = val_metrics.get('mse', 0)
                log_dict["val/mae"] = val_metrics.get('mae', 0)

        if wandb_cfg['use']:
            wandb.log(log_dict)

        # Print progress
        if dataset_cfg['task'] == 'zinc':
            print(f"epoch {epoch:03d} | train loss={tr_loss:.4f} mae={train_metrics.get('mae', 0):.4f} | "
                  f"val loss={va_loss:.4f} mae={val_metrics.get('mae', 0):.4f} | time {epoch_duration:.2f}s")
        elif dataset_cfg['task'] == 'shortest_path':
            print(f"epoch {epoch:03d} | train {tr_loss:.4f}/acc={tr_acc:.4f}/mse={train_metrics.get('mse', 0):.4f} | "
                  f"val {va_loss:.4f}/acc={va_acc:.4f}/mse={val_metrics.get('mse', 0):.4f} | time {epoch_duration:.2f}s")
        else:
            print(f"epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | time {epoch_duration:.2f}s")

        # For regression tasks (ZINC), track best model by MAE (lower is better)
        # For classification tasks, track by accuracy (higher is better)
        if dataset_cfg['task'] == 'zinc':
            val_mae = val_metrics.get('mae', float('inf'))
            if best_val == -1.0 or val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                time_to_best = time.time() - training_start_time
                torch.save(
                    {"state_dict": best_state, "config": config},
                    os.path.join(output_cfg['out_dir'], f"best_{output_cfg['run_name']}.pt"),
                )
        elif va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            time_to_best = time.time() - training_start_time
            torch.save(
                {"state_dict": best_state, "config": config},
                os.path.join(output_cfg['out_dir'], f"best_{output_cfg['run_name']}.pt"),
            )

    # Log total training time
    total_train_time = time.time() - training_start_time

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = eval_epoch(model.to(device), test_dl, crit, device, task=dataset_cfg['task'])
    te_loss = test_metrics['loss']

    # Print test results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Loss: {te_loss:.4f}")

    if dataset_cfg['task'] == 'zinc':
        # For ZINC regression, show MAE as primary metric
        print(f"MAE: {test_metrics.get('mae', 0):.4f}")
        print(f"MSE: {test_metrics.get('mse', 0):.4f}")
        print(f"RMSE: {test_metrics.get('rmse', 0):.4f}")
    else:
        # For classification tasks
        te_acc = test_metrics['accuracy']
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
        # Build test log based on task type
        if dataset_cfg['task'] == 'zinc':
            # Regression: log MAE, MSE, RMSE
            test_log = {
                "test/loss": te_loss,
                "test/mae": test_metrics.get('mae', 0),
                "test/mse": test_metrics.get('mse', 0),
                "test/rmse": test_metrics.get('rmse', 0),
                "time/total_train_time": total_train_time,
                "time/time_to_best_val": time_to_best,
            }
        else:
            # Classification: log accuracy, precision, recall, F1
            test_log = {
                "test/loss": te_loss,
                "test/acc": te_acc,
                "test/precision": test_metrics.get('precision', test_metrics.get('precision_macro', 0)),
                "test/recall": test_metrics.get('recall', test_metrics.get('recall_macro', 0)),
                "test/f1": test_metrics.get('f1', test_metrics.get('f1_macro', 0)),
                "time/total_train_time": total_train_time,
                "time/time_to_best_val": time_to_best,
            }
            # Add MSE/MAE for shortest_path
            if dataset_cfg['task'] == 'shortest_path':
                test_log["test/mse"] = test_metrics.get('mse', 0)
                test_log["test/mae"] = test_metrics.get('mae', 0)

        wandb.log(test_log)

        # Log confusion matrix as table
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
    ap = argparse.ArgumentParser(description="Train AGTT (AutoGraph tokenization + Transformer) on graph-token tasks")
    ap.add_argument("--config", type=str, default="configs/agtt_graph_token.yaml",
                    help="Path to YAML config file")
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Task: {config['dataset']['task']} | Algorithm: {config['dataset']['algorithm']}")
    print()

    main(config)
