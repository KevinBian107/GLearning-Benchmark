# Unified Query Embedding: IBTT vs AGTT

## Processing Flow

```
                    Same Native Graph
                    Edges: (0,1), (1,2), (2,3), (0,3)
                    Query: source=0, target=2
                              |
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
    IBTT Tokenization                  AGTT Tokenization
    (Index-based)                      (Trail-based)
            |                                   |
    <bos> 0 1 <e> 1 2 <e>              <sos> 0 1 2 3 0
    2 3 <e> 0 3 <e>
    <n> 0 1 2 3
            |                                   |
            |                                   |
            +----------------+------------------+
                             |
                             v
                    Append Query (SAME)
                             |
                        ... <q> 0 2
                             |
                             v
                    Embed: h = embed(x) + pos(ids)
                             |
                             v
                    Transformer: h = encoder(h)
                             |
                             v
            Extract Query Nodes (SAME LOGIC)
                             |
        Find <q> position in sequence
        u_emb = h[q_pos + 1]  (source)
        v_emb = h[q_pos + 2]  (target)
                             |
                             v
                  Concatenate (SAME)
                             |
            pooled = [graph_emb, u_emb, v_emb]
                             |
                             v
                    Classifier (SAME)
```

## Example Sequences After Query Append

**IBTT**:
```
<bos> 0 1 <e> 1 2 <e> 2 3 <e> 0 3 <e> <n> 0 1 2 3 <q> 0 2
                                                    ^  ^ ^
                                                    |  | |
                                                   <q> u v
                                                   pos 18,19,20
```

**AGTT**:
```
<sos> 0 1 2 3 0 <q> 0 2
                ^  ^ ^
                |  | |
               <q> u v
               pos 6,7,8
```

## Extraction (Same for Both)

```python
# Find <q> marker
q_pos = find_position_of_q_token(x)

# Extract at fixed offsets
u_emb = h[q_pos + 1]  # Source node embedding
v_emb = h[q_pos + 2]  # Target node embedding
```

## Key Point

**ONLY DIFFERENCE**: How graph structure is tokenized (before <q>)

**EVERYTHING ELSE IDENTICAL**: Query format, extraction logic, classification
