# Graph GPS

### [Refer to gps embedding method's documentation for querry encoding details](/docs/models/embed_ggps.md)

This experiment uses GraphGPS to test the ability of feeding synthetic native graphs into an modern graph transformer architecture like GraphGPS. It uses PyTorch Geometric's **GraphGym** framework, which implements a **registration pattern** to allow custom models, layers, and datasets to be plugged in without modifying the core framework code.

## 🔪 Sharp Bits: Registration Pattern

In train_ggps.py or GraphGPS/main.py we import `graphgps`, which triggers the registration chain

```
import graphgps
    │
    ├─> graphgps/__init__.py executes
    │   └─> from .network import *
    │
    ├─> graphgps/network/__init__.py executes
    │   └─> Sets __all__ = [all .py files in network/]
    │
    ├─> All network/*.py files get imported
    │
    ├─> graphgps/network/gps_model.py executes
    │   │
    │   └─> @register_network('GPSModel') decorator runs
    │       │
    │       └─> Executes: register.network_dict['GPSModel'] = GPSModel
    │
    ├─> graphgps/network/graphormer.py executes
    │   └─> register.network_dict['Graphormer'] = Graphormer
    │
    └─> ... (all other network files)
```

**Result:** A global dictionary `register.network_dict` now contains:
```python
{
    'GPSModel': <class GPSModel>,
    'Graphormer': <class Graphormer>,
    'BigBird': <class BigBird>,
    'SANTransformer': <class SANTransformer>,
    ...
}
```
Then in configurations like `configs/gps_graph_token.yaml`, we will look up what was registered, this is like gym from an RL environment; then teh following will directly be able to create the model:

```python
loaders = create_loader()
model = create_model()
```

This brings numerous benifits, namely:

1. **Modularity**: Add new models without modifying core code
2. **Discoverability**: All models are registered automatically when imported
3. **Configuration-driven**: Change models by editing YAML, not code
4. **Extensibility**: Easy to add custom models, heads, layers, optimizers, etc.

## Model Architecture:

GraphGPS (Graph Positional and Structural encoding with transformers) combines the strengths of Message Passing Neural Networks (MPNNs) and Transformers to learn on graph-structured data. The key innovation is using **global attention** alongside **local message passing** to capture both local neighborhood structure and long-range dependencies.

### Architecture Overview:

$$
h_0 = \text{Embed}(x) + \text{PE}(G)
$$
$$
h_{\ell} = \text{GPSLayer}(h_{\ell-1}, \text{edge\_index})
$$
$$
h_G = \text{GlobalPool}(h_L)
$$
$$
\hat{y} = \text{Head}(h_G)
$$

Where:
- **PE(G)**: Positional/structural encodings (e.g., Laplacian eigenvectors, random walk features)
- **GPSLayer**: Combines local MPNN + global attention
- **GlobalPool**: Graph-level aggregation (mean, sum, or attention-based)
- **Head**: Task-specific prediction head

---

### GPS Layer Architecture:

Each GPS layer implements a **hybrid local-global approach**:

$$
h'_i = h_i + \text{LocalMPNN}(h_i, \{h_j : j \in \mathcal{N}(i)\})
$$
$$
h''_i = h'_i + \text{GlobalAttn}(h'_i, \{h_j : j \in V\})
$$
$$
h^{(\ell)}_i = h''_i + \text{FFN}(h''_i)
$$

Where:
- **LocalMPNN**: Message passing over edges (e.g., GCN, GIN, GINE)
- **GlobalAttn**: Multi-head self-attention over all nodes
- **FFN**: Feed-forward network with residual connections

This design allows the model to:
1. Aggregate information from direct neighbors (local structure)
2. Attend to distant nodes (global patterns)
3. Learn both short-range and long-range dependencies

---

### Implementation Details:

#### 1. Node Feature Initialization
```python
# Embed raw node features + add positional encodings
self.node_encoder = nn.Linear(in_dim, hidden_dim)
self.pe_encoder = nn.Linear(pe_dim, hidden_dim)  # For Laplacian PE, RW PE, etc.

h = self.node_encoder(x) + self.pe_encoder(pe)
```

#### 2. GPS Layer Stack
```python
# Stack of L GPS layers
self.layers = nn.ModuleList([
    GPSLayer(
        dim_h=hidden_dim,
        local_gnn_type='GINEConv',      # Local MPNN: GCN, GIN, GINE, etc.
        global_model_type='Transformer', # Global attention
        num_heads=num_heads,
        dropout=dropout,
        attn_dropout=attn_dropout,
    )
    for _ in range(num_layers)
])
```

#### 3. Hybrid Local + Global Layer
```python
class GPSLayer(nn.Module):
    def forward(self, batch):
        h = batch.x

        # Local MPNN (message passing on edges)
        h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
        h = h + h_local  # Residual connection

        # Global Attention (full self-attention over nodes)
        h_attn = self.self_attn(h, batch.batch)  # batch indicates graph membership
        h = h + h_attn  # Residual connection

        # Feed-forward network
        h = h + self.ffn(h)

        batch.x = h
        return batch
```

#### 4. Graph-Level Pooling
```python
# Aggregate node embeddings to graph-level representation
if pooling == 'mean':
    h_graph = global_mean_pool(h, batch.batch)
elif pooling == 'add':
    h_graph = global_add_pool(h, batch.batch)
elif pooling == 'attention':
    h_graph = self.global_pool(h, batch.batch)
```

#### 5. Task-Specific Head
```python
# For graph classification (e.g., cycle detection)
self.head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_classes)
)
```

---

### Key Differences from Vanilla GNNs:

| Component | Vanilla GNN (e.g., GCN, GAT) | GraphGPS |
|-----------|------------------------------|----------|
| **Receptive field** | Local (k-hop neighbors) | Global (all nodes via attention) |
| **Long-range edges** | Requires many layers | Direct via self-attention |
| **Expressiveness** | Limited by message passing | Enhanced by global interactions |
| **Scalability** | O(E) per layer | O(N²) attention + O(E) MPNN |

---

### Training Objective:

For graph-level tasks (e.g., cycle detection, graph classification):
- **Loss**: Cross-entropy for classification, MSE for regression
  ```python
  loss = F.cross_entropy(logits, labels)
  ```
- **Optimizer**: AdamW with cosine learning rate schedule
- **Regularization**: Dropout, weight decay

For node-level tasks:
- Predict directly from node embeddings `h_i` without global pooling