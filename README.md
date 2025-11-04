# Graph Learning Benchmark
This is a repository for benchmarking graph learning methods's performance across various different tasks (for HDSI 2026 DSC180 Capstone).

## Environment
Refer to [this documentation](/docs/setup.md) for various different environmental setup.

## Running the Codebase

The [documentation](/docs/) folder contains information about models we are training here. Currently we supprt these two different methods for benchmarking performance:

| Aspect | GraphGPS (`train_ggps.py`) | Tokenize + Vanilla Transformer (`train_gtt.py`) |
|--------|----------------------------|--------------------------------------|
| **Input** | Native graph structure (nodes, edges) | Tokenized sequence |
| **Architecture** | GNN + Transformer hybrid | Pure Transformer encoder |
| **Inductive Bias** | Graph structure via message passing | Sequence order via positional encoding |
| **Pooling** | Graph pooling (mean/sum) | Sequence pooling (first token or mean) |
| **Attention** | Combines local (GNN) and global (Transformer) | Pure global self-attention |

### Graph Tokenization Transformer
First cd into the correct directory for generating synthetic graphs and tokenize them by following certain rule set:

```python
# auto setup enviornment + make graphs
bash graph_generator.sh

# auto setup enviornment + tokenize for tasks
bash task_generator.sh
```

This should automatically setup the environment for the graph-token repository and generate the training graphs and tasks. Then we need to switch the path name and the `split` argument in the `sh` file to `test` to generate the test directory. Then run the following to train an simple transformer for this task:

```bash
python train_gtt.py \
  --graph_token_root graph-token \
  --task cycle_check \
  --epochs 100 \
  --algorithm er \
  --run_name gtt-cycle-check \
  --wandb_project graph-token
  --out_dir runs_gtt
```

### Graph Native GPS
Similarly, we can run a GPS model upon the native graph like the following

```bash
python train_ggps.py \
  --graph_token_root graph-token \
  --task cycle_check \
  --algorithm er \
  --config configs/gps_graph_token.yaml \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --seed 0 \
  --use_wandb \
  --wandb_project graph-token \
  --run_name ggps-cycle-check \
  --out_dir runs_gps
```

### Graph Native Py-Geometric
At last, using similar setup as GPS, we can perform MPNN upon the native graph as well:

```bash
python train_mpnn.py \
    --graph_token_root graph-token \
    --task cycle_check \
    --algorithm er \
    --hidden_dim 128 \
    --num_layers 4 \
    --bs 64 \
    --lr 1e-3 \
    --epochs 100 \
    --run_name mpnn-cycle-check \
    --out_dir runs_mpnn \
    --wandb_project graph-token \
```

## Notebooks
- [IMBD & MUTAG GLearning Example](notebooks/simple_data.ipynb)
- [Comparing GCN & GAT Example](notebooks/gcn_gat.ipynb)