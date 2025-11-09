# Graph Tokenized Transformer

This experiment explores index-based tokenization of graphs using Ali Parviz’s [graph-token](https://github.com/alip67/graph-token) framework. Each graph is serialized into a sequence of discrete tokens, similar to how sentences are tokenized in NLP. A simple Transformer encoder model is trained to perform reasoning tasks on these sequences, such as detecting whether a graph contains a cycle.

Traditional GNNs reason over graph structure directly (edges/nodes). Here, the graph is flattened into a sequence so a Transformer can learn to infer structural properties from the ordering of tokens — analogous to learning syntax in language modeling. This approach tests how well sequence models like Transformers can generalize to graph reasoning tasks without explicit message-passing.

**We will be refering to both AutoGraph Tokenized Transformer `artt` and Index-Based Tokenized Transformer `ibtt` as Graph Tokenized Transformer`gtt` here***

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