# Training & Models
For each specific models, refer to [this model folder](/docs/models/) for more details. This documentation is an generic overview.

We have various different tasks at hand. Though we use the same model architecture, the input and output of these tasks are different (i.e. `cycle_check` is a graph-level binary classification while `shortest_path` is a node-pair-level multi-class classification or regression), hence requiring various different model input/output design. We summarized these particular designs in the following.

`cycle_check` and `shortest_path` differences:

| Aspect | `cycle_check` | `shortest_path` |
|--------|---------------|-----------------|
| **Task Type** | Graph-level binary classification | Node-pair-level multi-class classification |
| **Data Structure** | 1 graph → 1 sample | 1 graph → ~105 samples (multiple queries) |
| **Query Format** | `<q> has_cycle <p> yes/no` | `<q> shortest_distance u v <p> lenX` |
| **Label Format** | Binary: 0 (no) or 1 (yes) | Distance: 1, 2, 3, ..., 7 |
| **Output Classes** | 2 classes | 7+ classes (len1 through len7+) |
| **Data Fields** | text, label | text, label, query_u, query_v |

For graph Nnative models (MPNN, GPS, AGTT)

| Component | `cycle_check` | `shortest_path` |
|-----------|---------------|-----------------|
| **Input Features** | x = [1.0]` (constant) | x = [1.0, is_source, is_target] (with query encoding) |
| **Query Encoding** | N/A | Binary positional flags added via add_query_encoding_to_features(x, query_u, query_v) |
| **Forward Signature** | forward(data) | forward(data) where data.query_u, data.query_v exist |
| **Output Head** | Linear(hidden, 2) | Linear(hidden, num_classes) |
| **Key Change** | Standard graph classification | Add query flags to node features before GNN layers |

For tokenized sequence models (IBTT, AGTT)

| Component | `cycle_check` | `shortest_path` |
|-----------|---------------|-----------------|
| **Input Sequence** | Edges + nodes + query | Edges + nodes + query **(u, v included in text)** |
| **Label Stripping** | Strip <p> yes/no | Strip <p> lenX |
| **Query Handling** | Implicit (no specific nodes) | Explicit (u, v in sequence after shortest_distance) |
| **Output Head** | Linear(d_model, 2) | Linear(d_model, num_classes) |
| **Key Change** | Minimal - just output dimension | Change vocab parsing + output dimension |