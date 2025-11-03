# CLAUDE.md

A guide for benchmarking multiple graph learning methods on graph reasoning tasks using various input representations.
This is only a structure, not what have been implemented yet.

## 🎯 Project Overview

**Goal**: Benchmark different graph learning methods (vanilla MPNN, vanilla GAT, tokenization + vanilla transformer, GraphGPS, etc.) on graph reasoning tasks (cycle detection, connectivity, node degree, etc.) using different graph datasets.

**Key Insight**: Different methods consume different input formats from the same underlying graphs:
- **Graph-native methods** (MPNN, GAT, GraphGPS) → Take raw graph structure (nodes, edges, adjacency)
- **Sequence methods** (Transformer, Mamba) → Take tokenized graph sequences

## 🎨 Input Representation Philosophy

All methods are evaluated on the **same underlying graphs** but consume them in their natural format:
```
Synthetic Graph Generation (graph-token repo)
                ↓
        Generated Graphs
         /            \
        /              \
   Raw Graph          Tokenized Sequence
   (nodes, edges)     ("<bos> <n> 0 <e> 1 ...")
        ↓                      ↓
   Graph Methods          Sequence Methods
   - MPNN                 - Transformer
   - GAT                  - Mamba
   - GraphGPS             
```

### Why This Matters

- **Fair comparison**: Same graphs, different architectural inductive biases
- **Respect method design**: Each method uses its natural input format
- **Minimal adaptation**: Use each method's existing codebase/API
- **Clear insights**: Performance differences reflect architecture, not data preprocessing artifacts

## 📦 Method Categories

### 1. Graph-Native Methods (PyTorch Geometric)

**What they expect**: PyG `Data` objects with `x` (node features), `edge_index`, `y` (labels)

**Examples**:
- Vanilla MPNN (message passing)
- Vanilla GAT (graph attention)
- GraphGPS (graph transformer with virtual nodes)
- GIN (Graph Isomorphism Network)

**Integration approach**:
- Parse graph-token generated graphs into PyG format
- Use their existing training scripts with minimal config changes
- Keep their default hyperparameters as baseline
- Adapt only the dataset loading

### 2. Sequence-Based Methods

**What they expect**: Tokenized sequences with special tokens

**Examples**:
- Vanilla Transformer (your `train_gtt.py`)
- Mamba

**Integration approach**:
- Use graph-token's tokenized output directly
- Share `data_loader.py` across all sequence methods
- Write minimal training scripts for each architecture
- Keep sequence preprocessing consistent

## 🗂️ Data Pipeline

### Source: graph-token Repository

The `graph-token` repo generates synthetic graphs and provides **both** representations:
```python
# graph-token generates:
graph-token/tasks/cycle_check/er/
├── train/
│   ├── graph_0.json      # Contains BOTH representations
│   ├── graph_1.json
│   └── ...
└── test/

# Each JSON file contains:
{
    "nodes": [0, 1, 2, 3],           # Node list
    "edges": [[0,1], [1,2], [2,0]],  # Edge list
    "text": "<bos> <n> 0 <e> 1 <n> 1 <e> 2 <n> 2 <e> 0 <p> yes",
    "label": 1                        # Binary label (or continuous for regression)
}
```

### For Graph-Native Methods

Create a PyG dataset adapter:
```python
# graph_loaders/pyg_adapter.py
# Parses graph-token JSON → PyG Data objects
# Used by: MPNN, GAT, GraphGPS, etc.
```

Key considerations:
- **Node features**: Synthetic graphs may have no features → use one-hot node IDs or constant features
- **Edge features**: Most graph-token graphs are unweighted → binary edges
- **Task adaptation**: Binary classification (cycle detection) vs regression (node degree)

### For Sequence Methods

Use existing `data_loader.py`:
```python
# Already handles tokenized sequences
# Used by: Transformer, Mamba, LSTM, etc.
```

No changes needed - your current code already works for sequence methods.

## 🔧 Method Integration Strategy

### Respect Each Method's API

**Don't force a unified API**. Each method has its own training conventions:

#### Example: GraphGPS

GraphGPS uses Hydra configs and their custom training loop:
```bash
# GraphGPS/
cd GraphGPS/
conda activate graphgps

# Use THEIR config system
python main.py --cfg configs/GPS/graph-token-cycle.yaml

# Their config handles:
# - Model architecture (GPS layers, attention, virtual nodes)
# - Optimizer, scheduler
# - Logging, checkpointing
```

Your job: Write `GraphGPS/configs/GPS/graph-token-cycle.yaml` that points to your data

#### Example: Vanilla Transformer

You already have this working:
```bash
# Your existing train_gtt.py
python train_gtt.py \
    --task cycle_check \
    --algorithm er \
    --epochs 20
```

#### Example: Vanilla MPNN

Implement a simple training script using PyG:
```bash
# train_mpnn.py
python train_mpnn.py \
    --task cycle_check \
    --algorithm er \
    --hidden_dim 64 \
    --num_layers 3
```

## 📊 Experiment Configuration

### What to Control

**Fixed across methods**:
- Same train/val/test splits from graph-token
- Same random seeds for data generation
- Same evaluation metrics (accuracy for classification, MSE for regression)
- Same hardware (document GPU type)

**Method-specific** (use their defaults):
- Architecture hyperparameters (hidden dim, layers, etc.)
- Optimizer settings
- Learning rate schedules
- Regularization (dropout, weight decay)

### Why Use Their Defaults

- **Reproducibility**: Easier to compare with their published results
- **Fair comparison**: They chose hyperparameters that work well for their method
- **Less tuning**: Focus on comparing architectures, not hyperparameter search

You can note: "All methods use their respective default hyperparameters from original papers/repos"

## 🎯 Tasks & Datasets

### Available Tasks (from graph-token)
```
Classification Tasks:
- cycle_check: Does graph contain a cycle?
- connected_nodes: Are nodes u and v connected?
- edge_existence: Does edge (u,v) exist?

Regression Tasks:
- node_degree: What is degree of node u?
- node_count: How many nodes?
- edge_count: How many edges?
```

### Graph Generation Algorithms
```
- er: Erdős-Rényi (random)
- ba: Barabási-Albert (scale-free)
- sbm: Stochastic Block Model (communities)
- ws: Watts-Strogatz (small-world)
- grid: Grid graphs
```

### Experiment Matrix

For comprehensive benchmarking, run each method on all task-algorithm combinations:
```
Methods × Tasks × Algorithms
(5 methods) × (6 tasks) × (3-5 algorithms) = 90-150 experiments
```

Start with a subset: `cycle_check` on `er`, `ba`, `sbm` for all methods

## 🧪 Reproducibility Guidelines

### For Each Method
```bash
# 1. Environment
conda env export > conda_envs/{method_name}.yml

# 2. Seeds
Set seed in: data generation, train/val split, model initialization, training

# 3. Logging
Log to W&B with tags: method, task, algorithm, seed

# 4. Checkpoints
Save: best validation model, final model, training config

# 5. Metrics
Report: train/val/test accuracy, loss, training time, # parameters
```

### Comparison Protocol

Run each method **3-5 times** with different seeds:
- Report mean ± std of test accuracy
- Compare training time and memory usage
- Analyze failure cases (which graphs are hard for which methods?)

## 🐛 Common Gotchas

### Graph-Native Methods

**Empty node features**: Synthetic graphs may have no node attributes
- Solution: Use one-hot node IDs, or constant 1s, or positional encodings

**Batch size differences**: PyG batches graphs differently than sequence batches
- Solution: Use PyG's DataLoader, tune batch size separately

**Different tasks need different output heads**:
- Binary classification: 2-class softmax
- Node pair query: Concatenate node embeddings → classifier
- Regression: Linear output head

### Sequence Methods

**Variable length sequences**: Different graphs → different token lengths
- Solution: Already handled by your `collate()` function with padding

**Label stripping**: Don't feed the answer to the model during training
- Solution: Your `TokenDataset(strip_label=True)` already handles this

**Context window**: Very large graphs may exceed max sequence length
- Solution: Truncate or use hierarchical tokenization

## 📝 Documentation for Each Method

Create a brief README in each method's directory:
```markdown
# Method: GraphGPS

## Environment
conda activate graphgps

## Data Format
PyG Data objects with:
- x: node features (one-hot node IDs)
- edge_index: COO format edges
- y: binary labels

## Running
python main.py --cfg configs/graph-token-cycle.yaml

## Expected Performance
- cycle_check (ER): ~96% test accuracy
- Training time: ~15min on V100

## Notes
- Uses virtual node for global pooling
- Default: 4 GPS layers, 8 attention heads
- No hyperparameter tuning done
```

## 🚀 Getting Started

1. **Verify graph-token data generation works**
```bash
   cd graph-token/
   python generate.py --task cycle_check --algorithm er
   # Verify both graph structure and tokenization are in JSON
```

2. **Run baseline transformer** (you already have this)
```bash
   python train_gtt.py --task cycle_check --algorithm er
```

---

**Remember**: The goal is to compare architectural inductive biases fairly, not to find the "best hyperparameters for each method on this specific task." Use defaults, focus on clear comparisons.