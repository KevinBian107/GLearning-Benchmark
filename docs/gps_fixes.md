# GraphGPS Training Issues
Two critical bugs were identified and resolved in the GraphGPS training pipeline when attempting to train on the shortest path prediction task:

1. **Query Encoding Batching Issue**: Query node encodings were incorrectly applied to batched graphs, causing only the first graph's query nodes to be used for all graphs in a batch.
2. **Edge Attribute Dimension Mismatch**: Incompatibility between dataset-generated edge attributes and the GINE layer's expectations.


## Issue #1: Query Encoding Batching Problem

### Problem Description

```bas
ValueError: Node and edge feature dimensionalities do not match.
Consider setting the 'edge_dim' attribute of 'GINEConv'
```

The original implementation attempted to add query node encodings during the model's forward pass:

```python
# trainer/train_ggps.py - BROKEN CODE
class GPSWrapper(nn.Module):
    def forward(self, batch):
        if self.task == 'shortest_path' and hasattr(batch, 'query_u'):
            # PROBLEM: Uses only first graph's query nodes
            query_u = batch.query_u[0].item()
            query_v = batch.query_v[0].item()
            batch.x = add_query_encoding_to_features(batch.x, query_u, query_v)

        out = self.gps_model(batch)
        return out
```

1. **Batching in PyTorch Geometric**: When multiple graphs are batched (e.g., batch_size=128), PyG concatenates them into a single large graph:
   ```
   Graph 1: nodes [0,1,2,3,4,5,6]     query: 2→6
   Graph 2: nodes [7,8,...,23]        query: 10→15
   Graph 3: nodes [24,25,...,38]      query: 30→35
   ...
   Batched: nodes [0,1,2,...,N_total]
   ```

2. **Per-Graph Metadata Lost**: Each graph has different query nodes, but `batch.query_u[0]` only retrieves the first graph's query nodes.

3. **Incorrect Encoding**: The code was marking nodes 2 and 6 as source/target for **all 128 graphs**, when it should mark different nodes for each graph.

4. **Dimension Inconsistency**: Attempting to concatenate features on a batched tensor without proper indexing caused dimension mismatches.

### Solution

**Approach**: Move query encoding from forward pass to dataset preprocessing using PyG transforms.


**Step 1**: Create a PyG transform class

```python
# graph_data_loader/graph_token_dataset.py
class AddQueryEncoding:
    """Transform that adds query node encoding to node features."""

    def __call__(self, data):
        """Apply to individual graph BEFORE batching."""
        if hasattr(data, 'query_u') and hasattr(data, 'query_v'):
            query_u = data.query_u.item()
            query_v = data.query_v.item()
            data.x = add_query_encoding_to_features(data.x, query_u, query_v)
        return data
```

**Step 2**: Apply transform during dataset creation

```python
# trainer/train_ggps.py
pre_transform = AddQueryEncoding() if task == 'shortest_path' else None

train_dataset = GraphTokenDataset(
    root=graph_token_root,
    task=task,
    algorithm=algorithm,
    split='train',
    pre_transform=pre_transform,  # ✅ Applied per-graph before batching
)
```

**Step 3**: Remove forward-pass encoding

```python
# trainer/train_ggps.py
class GPSWrapper(nn.Module):
    def forward(self, batch):
        # Query encoding already in batch.x from dataset transform
        out = self.gps_model(batch)
        return out
```

## Issue #2: Edge Attribute Dimension Mismatch

Initial error after removing `edge_attr`:
```bash
AttributeError: 'NoneType' object has no attribute 'size'
```

Previous error with 64-dim `edge_attr`:
```python
ValueError: Node and edge feature dimensionalities do not match.
Consider setting the 'edge_dim' attribute of 'GINEConv'
```

**Mismatch between dataset and model configuration:**

1. **Dataset** was creating edge attributes:
   ```python
   # graph_token_dataset.py - BROKEN CODE
   edge_attr = torch.ones((num_edges, 64), dtype=torch.float)
   ```

2. **GPS Config** was using GINE (requires edge features):
   ```yaml
   # configs/gps_graph_token.yaml - BROKEN CONFIG
   gt:
     layer_type: GINE+Transformer  # GINE = GIN with Edge features
   ```

3. **GINE Layer** expects edge features to match node features after encoding:
   - Node features: 1-dim → encoded to 32-dim
   - Edge features: 64-dim → no encoder configured
   - **64 ≠ 32** → Dimension mismatch!

4. **Graph-Token Graphs**: These synthetic graphs are unweighted and don't need edge features.

### Solution

**Approach**: Use GIN (without edge features) instead of GINE, and remove edge attribute generation.


**Step 1**: Remove edge attributes from dataset

```python
# graph_data_loader/graph_token_dataset.py
data = Data(
    x=x,
    edge_index=edge_index,
    # edge_attr removed - not needed for unweighted graphs
    y=torch.tensor([label], dtype=torch.long),
    num_nodes=num_nodes,
)
```

**Step 2**: Change GPS configuration to use GIN

```yaml
# configs/gps_graph_token.yaml
gt:
  layer_type: GIN+Transformer  # GIN instead of GINE
  layers: 2
  n_heads: 4
  dim_hidden: 32
```