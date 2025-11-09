# Graph Learning Benchmark
This is a repository for benchmarking graph learning methods's performance across various different tasks (for HDSI 2026 DSC180 Capstone).

## Environment
Refer to [this documentation](/docs/setup.md) for various different environmental setup.

## Running the Codebase

The [documentation](/docs/) folder contains information about models we are training here. Currently we support four different methods for benchmarking performance:

| Aspect / Component | GraphGPS (`train_ggps.py`) | Index-Based Tokenization + Transformer (`train_ibtt.py`) | Trail-Based Tokenization + Transformer (`train_agtt.py`) | MPNN (`train_mpnn.py`) |
|------------------|---------------------------|-------------------------------------------------------|--------------------------------------------------------|-----------------------|
| **Input** | Native graph | Pre-tokenized sequence | Native graph → Trail tokenization | Native graph |
| **Architecture** | GNN + Transformer hybrid | Pure Transformer encoder | Pure Transformer encoder | Message Passing Neural Network (e.g., GIN) |
| **Tokenization** | None (direct GNN) | Index-based (graph-token) | Trail-based (AutoGraph SENT) | None (direct GNN) |
| **Inductive Bias** | Graph topology + global attention | Sequence order via positional encoding | Sequence order + trail structure | Graph topology via message passing |
| **Attention Scope** | Local (GNN) + Global (self-attn) | Global self-attention | Global self-attention | Local (k-hop neighborhoods) |
| **Pooling** | Graph pooling (mean/sum) | Sequence pooling (first token or mean) | Sequence pooling (first token) | Graph pooling (mean/sum) |
| **Complexity** | O(E) + O(N²) | O(N²) | O(N²) | O(E) per layer |

### Configurations

All training scripts use YAML configuration files for reproducibility and ease of use. Each method has a default config file in the `configs/` directory:

- **MPNN**: `configs/mpnn_graph_token.yaml`
- **IBTT (Index-Based Tokenization + Transformer)**: `configs/ibtt_graph_token.yaml`
- **AGTT (AutoGraph Trail Tokenization + Transformer)**: `configs/agtt_graph_token.yaml`
- **GraphGPS**: `configs/gps_graph_token.yaml`

Config Structure: Each config file contains sections for:
- `dataset` or `data`: Task, algorithm, and data paths
- `model`: Architecture hyperparameters
- `train`: Training settings (batch size, epochs, learning rate)
- `output`: Output directory and run naming
- `wandb`: Weights & Biases logging configuration

---

### Index-Based Tokenization Transformer (IBTT)
First cd into the correct directory for generating synthetic graphs and tokenize them by following certain rule set:

```bash
# auto setup enviornment + make graphs
bash graph_generator.sh

# auto setup enviornment + tokenize for tasks
bash task_generator.sh
```

This should automatically setup the environment for the graph-token repository and generate the training graphs and tasks. Then we need to switch the path name and the `split` argument in the `sh` file to `test` to generate the test directory. Then run the following to train a simple transformer for this task:

```bash
conda activate glearning_180a
python train_ibtt.py
```

### AutoGraph Trail Tokenization Transformer (AGTT)
This method uses AutoGraph's trail-based tokenization (SENT algorithm) to convert native graphs into sequences, then trains a vanilla transformer. This allows direct comparison with IBTT to isolate the effect of tokenization strategy:

```bash
conda activate autograph
python train_agtt.py
```

**Note**: AGTT uses the same transformer architecture as IBTT but with different tokenization. It loads native graphs (like MPNN/GraphGPS) and applies AutoGraph's trail-based tokenization instead of index-based tokenization. This enables comparing **tokenization strategies** while keeping the model architecture constant.


### Graph Native GPS
Similarly, we can run a GPS model upon the native graph like the following:

```bash
conda activate graphgps
python train_ggps.py
```

**Note**: GraphGPS uses a more complex config structure with sections like `gt` (graph transformer), `gnn`, and `optim`. See [docs/ggps.md](docs/ggps.md) for details on the architecture.


### Graph Native MPNN (Message Passing Neural Network)
At last, using similar setup as GPS, we can perform MPNN upon the native graph as well:

```bash
conda activate glearning_180a
python train_mpnn.py
```

**Note**: Our MPNN implementation uses GIN (Graph Isomorphism Network) layers, which are provably as expressive as the Weisfeiler-Leman graph isomorphism test. See [docs/mpnn.md](docs/mpnn.md) for detailed architecture information.


## Notebooks
- [IMBD & MUTAG GLearning Example](notebooks/simple_data.ipynb)
- [Comparing GCN & GAT Example](notebooks/gcn_gat.ipynb)


## Acknowledgements
- Official [graph-token repository](https://github.com/alip67/graph-token)
- Official [Autograph repository](https://github.com/BorgwardtLab/AutoGraph)
- Official [GraphGPS repositroy](https://github.com/rampasek/GraphGPS)
- Official [Py-Geomeric documentation](https://pytorch-geometric.readthedocs.io/en/latest/)