# Graph Learning Benchmark
This is a repository for benchmarking graph learning methods's performance across various different tasks (for HDSI 2026 DSC180 Capstone).

## Environment
Refer to [this documentation](/docs/setup.md) for various different environmental setup.

## Running the Codebase

The [documentation](/docs/) folder contains information about models we are training here. Currently we support three different methods for benchmarking performance:

| Aspect / Component | GraphGPS (`train_ggps.py`) | Tokenize + Vanilla Transformer (`train_gtt.py`) | MPNN (`train_mpnn.py`) |
|------------------|---------------------------|-----------------------------------------------|-----------------------|
| **Input** | Native graph | Tokenized sequence | Native graph |
| **Architecture** | GNN + Transformer hybrid | Pure Transformer encoder | Message Passing Neural Network (e.g., GIN) |
| **Inductive Bias** | Graph topology + global attention | Sequence order via positional encoding | Graph topology via message passing |
| **Attention Scope** | Local (GNN) + Global (self-attn) | Global self-attention | Local (k-hop neighborhoods) |
| **Pooling** | Graph pooling (mean/sum) | Sequence pooling (first token or mean) | Graph pooling (mean/sum) |
| **Complexity** | O(E) + O(N²) | O(N²) | O(E) per layer |

### Configurations

All training scripts use YAML configuration files for reproducibility and ease of use. Each method has a default config file in the `configs/` directory:

- **MPNN**: `configs/mpnn_graph_token.yaml`
- **GTT (Graph Tokenization + Transformer)**: `configs/gtt_graph_token.yaml`
- **GraphGPS**: `configs/gps_graph_token.yaml`

Config Structure: Each config file contains sections for:
- `dataset` or `data`: Task, algorithm, and data paths
- `model`: Architecture hyperparameters
- `train`: Training settings (batch size, epochs, learning rate)
- `output`: Output directory and run naming
- `wandb`: Weights & Biases logging configuration

---

### Graph Tokenization Transformer
First cd into the correct directory for generating synthetic graphs and tokenize them by following certain rule set:

```bash
# auto setup enviornment + make graphs
bash graph_generator.sh

# auto setup enviornment + tokenize for tasks
bash task_generator.sh
```

This should automatically setup the environment for the graph-token repository and generate the training graphs and tasks. Then we need to switch the path name and the `split` argument in the `sh` file to `test` to generate the test directory. Then run the following to train a simple transformer for this task:

```bash
python train_gtt.py
python train_gtt.py --config configs/gtt_graph_token.yaml
```

**Configuration**: Edit `configs/gtt_graph_token.yaml` to customize:
- Dataset (task, algorithm, data paths)
- Model architecture (d_model, nhead, nlayers, dropout)
- Training parameters (batch_size, epochs, learning rate)
- Output settings and W&B logging

### Graph Native GPS
Similarly, we can run a GPS model upon the native graph like the following:

```bash
python train_ggps.py
python train_ggps.py --config configs/gps_graph_token.yaml
```

**Configuration**: Edit `configs/gps_graph_token.yaml` to customize:
- Dataset (task, algorithm, data paths)
- Model architecture (GPS layers, attention heads, hidden dimensions)
- Training parameters (batch_size, epochs, learning rate, scheduler)
- Output settings and W&B logging

**Note**: GraphGPS uses a more complex config structure with sections like `gt` (graph transformer), `gnn`, and `optim`. See [docs/ggps.md](docs/ggps.md) for details on the architecture.


### Graph Native MPNN (Message Passing Neural Network)
At last, using similar setup as GPS, we can perform MPNN upon the native graph as well:

```bash
python train_mpnn.py
python train_mpnn.py --config configs/mpnn_graph_token.yaml
```

**Configuration**: Edit `configs/mpnn_graph_token.yaml` to customize:
- Dataset (task, algorithm, data paths)
- Model architecture (hidden_dim, num_layers, dropout, pooling strategy)
- Training parameters (batch_size, epochs, learning rate)
- Output settings and W&B logging

**Note**: Our MPNN implementation uses GIN (Graph Isomorphism Network) layers, which are provably as expressive as the Weisfeiler-Leman graph isomorphism test. See [docs/mpnn.md](docs/mpnn.md) for detailed architecture information.


## Notebooks
- [IMBD & MUTAG GLearning Example](notebooks/simple_data.ipynb)
- [Comparing GCN & GAT Example](notebooks/gcn_gat.ipynb)


## Acknowledgements
- Official [graph-token repository](https://github.com/alip67/graph-token)
- Official [Autograph repository](https://github.com/BorgwardtLab/AutoGraph)
- Official [GraphGPS repositroy](https://github.com/rampasek/GraphGPS)