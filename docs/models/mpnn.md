# Message Passing Neural Networks

This experiment uses a vanilla Message Passing Neural Network (MPNN) based on the **Graph Isomorphism Network (GIN)** architecture to test the ability of graph neural networks to learn on synthetic graph reasoning tasks. Unlike sequence-based methods or hybrid architectures, MPNNs work directly with graph structure using local message passing.

### The Message Passing Framework

Message Passing Neural Networks form the foundation of modern graph learning. The core idea is simple yet powerful: **nodes iteratively aggregate information from their neighbors** to build increasingly expressive representations. At each layer $\ell$, each node $i$ performs three steps:

1. **Message Creation**: Neighbors $j \in \mathcal{N}(i)$ send messages based on their features
2. **Aggregation**: Node $i$ collects and combines messages from all neighbors
3. **Update**: Node $i$ updates its representation using aggregated messages

Formally:

$$
\text{message}_{i \leftarrow j}^{(\ell)} = \text{MSG}^{(\ell)}(h_i^{(\ell-1)}, h_j^{(\ell-1)}, e_{ij})
$$

$$
\text{aggregate}_i^{(\ell)} = \text{AGG}^{(\ell)}(\{\text{message}_{i \leftarrow j}^{(\ell)} : j \in \mathcal{N}(i)\})
$$

$$
h_i^{(\ell)} = \text{UPDATE}^{(\ell)}(h_i^{(\ell-1)}, \text{aggregate}_i^{(\ell)})
$$

Where:
- $h_i^{(\ell)}$: Hidden representation of node $i$ at layer $\ell$
- $e_{ij}$: Edge features (if available)
- $\mathcal{N}(i)$: Neighborhood of node $i$

---

## Graph Isomorphism Networks (GIN)

Our implementation uses **GIN**, which is provably as powerful as the Weisfeiler-Leman (WL) graph isomorphism test. GIN was designed to maximize the expressive power of message passing by using injective aggregation functions.

- **Problem with simpler GNNs**: Methods like GCN or GraphSAGE use mean/max pooling, which can map different neighbor sets to the same representation (they're not injective).

- **GIN's solution**: Use a **sum aggregation** (provably injective for multisets) combined with learnable MLPs:

$$
h_i^{(\ell)} = \text{MLP}^{(\ell)} \left( (1 + \epsilon^{(\ell)}) \cdot h_i^{(\ell-1)} + \sum_{j \in \mathcal{N}(i)} h_j^{(\ell-1)} \right)
$$

Where:
- **Sum aggregation**: Preserves multiset information (unlike mean/max)
- **$\epsilon$**: Learnable or fixed scalar to weight the central node
- **MLP**: Multi-layer perceptron for non-linear transformation

**Key insight**: By Theorem 3 from the GIN paper (Xu et al., 2019), this formulation can distinguish any graphs that the WL test can distinguish.

---

## Model Architecture

Our MPNN implementation follows this structure:

### 1. Node Feature Initialization

$$
h_i^{(0)} = \text{Embed}(x_i)
$$

For synthetic graphs with no features:
```python
# Constant features (all nodes have feature [1.0])
x = torch.ones((num_nodes, 1), dtype=torch.float)

# Or one-hot encoding of node IDs
x = F.one_hot(torch.arange(num_nodes), num_nodes).float()
```

### 2. GIN Convolution Layers

Each GIN layer consists of:

$$
h_i^{(\ell)} = \text{BN}\left(\text{ReLU}\left(\text{MLP}^{(\ell)} \left( (1 + \epsilon) \cdot h_i^{(\ell-1)} + \sum_{j \in \mathcal{N}(i)} h_j^{(\ell-1)} \right)\right)\right)
$$

The MLP is a 2-layer feedforward network:
$$
\text{MLP}(x) = W_2 \cdot \text{Dropout}(\text{ReLU}(\text{BN}(W_1 x + b_1))) + b_2
$$

### 3. Graph-Level Pooling

After $L$ layers of message passing, aggregate node embeddings to a graph-level representation:

$$
h_G = \text{POOL}(\{h_i^{(L)} : i \in V\})
$$

**Pooling options**:
- **Mean**: $h_G = \frac{1}{|V|} \sum_{i \in V} h_i^{(L)}$ (used in our implementation)
- **Sum**: $h_G = \sum_{i \in V} h_i^{(L)}$ (preserves graph size information)
- **Max**: $h_G = \max_{i \in V} h_i^{(L)}$ (selects most salient features)

### 4. Classification Head

$$
\hat{y} = \text{Softmax}(W_{\text{cls}} \cdot h_G + b_{\text{cls}})
$$

For binary classification (e.g., cycle detection):
- Output dimension: 2 (yes/no)
- Loss: Cross-entropy

    $$
    \mathcal{L} = -\frac{1}{|B|} \sum_{G \in B} \sum_{c=1}^{C} y_{G,c} \log(\hat{y}_{G,c})
    $$

    Where:
    - $B$: Batch of graphs
    - $C$: Number of classes (2 for binary classification)
    - $y_{G,c}$: True label for graph $G$, class $c$
    - $\hat{y}_{G,c}$: Predicted probability

---

## Key Design Choices for Synthetic Graphs

| Aspect | GCN | GAT | GIN |
|--------|-----|-----|-----|
| **Aggregation** | Normalized mean | Weighted mean (attention) | Sum |
| **Expressiveness** | Limited (WL-test) | Limited (WL-test) | Maximal (WL-test) |
| **Injectivity** | ❌ Not injective | ❌ Not injective | ✅ Injective |
| **Best for** | Node features matter | Heterogeneous graphs | Graph structure reasoning |

For synthetic graph reasoning tasks (cycle detection, connectivity), **structure matters more than node features**, making GIN the ideal choice.

### Node Features for Synthetic Graphs

Synthetic graphs often have no meaningful node features. Options:

**a) Constant features (our choice)**:
```python
x = torch.ones((num_nodes, 1))
```
- **Pros**: Simple, forces the model to learn from structure only
- **Cons**: All nodes start identical

**b) One-hot node IDs**:
```python
x = F.one_hot(torch.arange(num_nodes), num_nodes)
```
- **Pros**: Distinguishes nodes initially
- **Cons**: Doesn't generalize to graphs of different sizes

**c) Positional encodings**:
```python
x = laplacian_positional_encoding(graph, k=10)
```
- **Pros**: Encodes structural position
- **Cons**: Expensive to compute, may leak information

### Batch Normalization

Applied after each GIN layer to:
- Stabilize training (normalize activations)
- Allow higher learning rates
- Reduce internal covariate shift

---