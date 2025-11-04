# DSC180-Benchmarks
Practice Codebase for DSC180 Capstone Graph Learning & Graph Tokenization. The [documentation](/docs/) folder contains information about models we are training here.

## Environment
Refer to [this documentation](/docs/setup.md) for various different environmental setup.

## Running the Codebase

Currently we supprt these two different methods for benchmarking performance:

| Aspect | GraphGPS (`train_ggps.py`) | Vanilla Transformer (`train_gtt.py`) |
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
  --algorithm er \
  --run_name gtt-cycle-check \
  --wandb_project graph-token
```

### Graph Native GPS
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
  --wandb_project graph-token-gps \
  --run_name gps-cycle-er-seed0 \
  --out_dir runs_gps
```

### Graph Basic Py-geometric
- [IMBD & MUTAG GLearning Example](notebooks/simple_data.ipynb)
- [Comparing GCN & GAT Example](notebooks/gcn_gat.ipynb)
