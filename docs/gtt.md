# Graph Tokenized Transformer

This experiment explores index-based tokenization of graphs using Ali Parviz’s [graph-token](https://github.com/alip67/graph-token) framework. Each graph is serialized into a sequence of discrete tokens, similar to how sentences are tokenized in NLP. A simple Transformer encoder model is trained to perform reasoning tasks on these sequences, such as detecting whether a graph contains a cycle.

Traditional GNNs reason over graph structure directly (edges/nodes). Here, the graph is flattened into a sequence so a Transformer can learn to infer structural properties from the ordering of tokens — analogous to learning syntax in language modeling. This approach tests how well sequence models like Transformers can generalize to graph reasoning tasks without explicit message-passing.

---

## Prediction Tasks:
### CycleCheck:
The model’s objective is to predict whether the given tokenized graph encodes a cyclic or acyclic structure.

- **Input:** A graph \( G = (V, E) \) serialized into a token sequence:
  ```
  <bos> <n> 0 1 2 3 <e> 0 1 1 2 2 3 ... <q> <p> yes/no <eos>
  ```
  - `<n>` introduces the list of nodes.
  - `<e>` introduces the list of edges.
  - `<q>` marks the query.
  - `<p>` is followed by the target label.
- **Output:** A binary label:
  - `yes` → graph contains at least one cycle  
  - `no`  → graph is acyclic

### EdgeExistence:
The model’s objective is to determine whether a given pair of nodes is directly connected by an edge.

- **Input:** A graph \( G = (V, E) \) serialized into a token sequence:
  ```
  <bos> <n> 0 1 2 3 <e> 0 1 1 2 2 3 ... <q> 0 3 <p> yes/no <eos>
  ```
  - `<n>` introduces the list of nodes.
  - `<e>` introduces the list of edges.
  - `<q>` introduces the query — a pair of node indices (e.g., `0 3`).
  - `<p>` marks the prediction token, followed by the answer.
- **Output:** A binary label:
  - `yes` → the queried nodes share an edge  
  - `no`  → the queried nodes are not directly connected

### ConnectedNodes:
The model’s objective is to predict whether two nodes are connected by any path (not necessarily direct).

- **Input:** A graph \( G = (V, E) \) serialized into a token sequence:
  ```
  <bos> <n> 0 1 2 3 <e> 0 1 1 2 2 3 ... <q> 0 3 <p> yes/no <eos>
  ```
  - `<n>` introduces the list of nodes.
  - `<e>` introduces the list of edges.
  - `<q>` introduces the query — the node pair to check for reachability.
  - `<p>` marks the prediction label position.
- **Output:** A binary label:
  - `yes` → there exists a path between the queried nodes  
  - `no`  → the nodes belong to disconnected components


### NodeCount:
The model’s objective is to predict the total number of nodes in the graph.

- **Input:** A graph \( G = (V, E) \) serialized into a token sequence:
  ```
  <bos> <n> 0 1 2 3 4 5 <e> 0 1 1 2 2 3 ... <q> <p> N <eos>
  ```
  - `<n>` introduces all nodes.
  - `<e>` introduces the list of edges.
  - `<q>` marks the start of the query.
  - `<p>` marks the predicted output token, followed by the numeric label.
- **Output:** An integer label:
  - The number of distinct nodes \( |V| \) in the graph


### EdgeCount:
The model’s objective is to predict the total number of edges in the graph.

- **Input:** A graph \( G = (V, E) \) serialized into a token sequence:
  ```
  <bos> <n> 0 1 2 3 <e> 0 1 1 2 2 3 ... <q> <p> M <eos>
  ```
  - `<n>` introduces the list of nodes.
  - `<e>` lists all edges.
  - `<q>` marks the query.
  - `<p>` precedes the numeric label.
- **Output:** An integer label:
  - The number of edges \( |E| \) in the graph

### NodeDegree:
The model’s objective is to predict the degree (number of incident edges) of a specific node.

- **Input:** A graph \( G = (V, E) \) serialized into a token sequence:
  ```
  <bos> <n> 0 1 2 3 <e> 0 1 1 2 2 3 ... <q> 1 <p> D <eos>
  ```
  - `<n>` introduces the nodes.
  - `<e>` lists all edges.
  - `<q>` introduces the node whose degree should be predicted.
  - `<p>` precedes the numeric answer.
- **Output:** An integer label:
  - The degree of the queried node (number of connected edges)

---

## Model Architecture:
The lightweight encoder-only Transformer architecture:

$$
h_0 = \text{Embed}(x) + \text{PosEmbed}(x)
$$
$$
H = \text{TransformerEncoder}(h_0)
$$
$$
z = \text{Pool}(H)
$$
$$
\hat{y} = \text{softmax}(Wz + b)
$$


- Embedding layer → maps tokens to vectors.
  ```python
  self.embed = nn.Embedding(vocab_size, d_model)
  ```
- Positional embeddings → inject the absolute order information (note that this is not a graph positional encoding, it is the encoding for the sequence itself).
  ```python
  self.pos = nn.Embedding(max_pos, d_model)
  ```

- TransformerEncoder (2–6 layers, multi-head self-attention).
  ```python
  enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=p_drop,
            batch_first=True,
        )
  ```

- Pooling: either the hidden state at `<bos>` or the mean of all tokens.
  ```python
  pooled = (h * attn_mask.unsqueeze(-1)).sum(1) / lens
  ```

- Classifier head: a linear layer mapping to 2 logits.
  ```python
  self.cls = nn.Linear(d_model, 2)
  ```

The training objective:
- Loss: Cross-entropy between predicted logits and binary label.
- Optimizer: AdamW with gradient clipping.