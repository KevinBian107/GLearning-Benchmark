# Tokenization: Query Node Embedding

## Complete Data Flow

### Input Example
```
Text: "<bos> 0 1 <e> 1 2 <e> 2 3 <e> <n> 0 1 2 3 <q> shortest_distance 0 3 <p>"
Label: len3 (distance from node 0 to node 3 is 3)
```

---

## BEFORE: Single Embedding Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Input Sequence (after stripping label)                     │
│ <bos> 0 1 <e> 1 2 <e> 2 3 <e> <n> 0 1 2 3 <q> SD 0 3 <p>   │
└────────────────────────┬────────────────────────────────────┘
                         │
                    Vocabulary
                    Mapping
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Token IDs                                                   │
│ [1, 9, 10, 2, 10, 11, 2, 11, 12, 2, 3, 9, 10, 11, 12, 4,   │
│  50, 9, 12, 5]                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                  Embedding Lookup
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Token Embeddings (each is d_model=16 dimensional)          │
│                                                             │
│ Position 0:  [0.03, 0.01, -0.04, ...]  ← <bos>            │
│ Position 1:  [-0.01, 0.04, -0.02, ...] ← "0"              │
│ Position 2:  [0.05, -0.03, 0.01, ...]  ← "1"              │
│ ...                                                         │
│ Position 17: [-0.01, 0.04, -0.02, ...] ← "0" (query!)     │
│ Position 18: [0.02, -0.05, 0.03, ...]  ← "3" (query!)     │
│ Position 19: [0.01, 0.02, -0.01, ...]  ← <p>              │
└────────────────────────┬────────────────────────────────────┘
                         │
              + Positional Encodings
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Transformer (self-attention, FFN, ...)                     │
│                                                             │
│ Updates all embeddings based on context                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Output Embeddings h                                         │
│                                                             │
│ h[0]:  [0.12, -0.05, 0.08, ...]  ← <bos> (updated)        │
│ h[1]:  [0.07, 0.09, -0.03, ...]  ← "0" (updated)           │
│ h[2]:  [-0.02, 0.11, 0.04, ...]  ← "1" (updated)           │
│ ...                                                         │
│ h[17]: [0.05, 0.08, -0.02, ...]  ← "0" query (updated)    │
│ h[18]: [-0.03, 0.06, 0.07, ...]  ← "3" query (updated) ✗  │
│ h[19]: [0.04, -0.01, 0.02, ...]  ← <p> (updated)           │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴──────────────┐
            │  PROBLEM: Only use h[0]! │
            │  Throw away h[1..19]     │
            └────────────┬──────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Pooling: pooled = h[0]                                      │
│                                                             │
│ [0.12, -0.05, 0.08, ..., 0.03]                             │
│ Shape: (16,) ← All info in just 16 numbers!                │
└────────────────────────┬────────────────────────────────────┘
                         │
                  Layer Norm
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Classifier: Linear(16 → 7)                                  │
│                                                             │
│ [2.3, -1.5, 0.8, -0.3, 1.2, -0.7, 0.4]                     │
│  len1  len2  len3  len4  len5  len6  len7                  │
└─────────────────────────────────────────────────────────────┘
```

**Bottleneck**: 16-dimensional vector encodes everything!

---

## AFTER: Three Embedding Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Input Sequence (same as before)                            │
│ <bos> 0 1 <e> 1 2 <e> 2 3 <e> <n> 0 1 2 3 <q> SD 0 3 <p>   │
└────────────────────────┬────────────────────────────────────┘
                         │
              (Vocabulary, Embedding, Transformer - same)
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Output Embeddings h (same as before)                        │
│                                                             │
│ h[0]:  [0.12, -0.05, 0.08, ...]  ← <bos>                  │
│ h[1]:  [0.07, 0.09, -0.03, ...]  ← "0"                     │
│ ...                                                         │
│ h[17]: [0.05, 0.08, -0.02, ...]  ← "0" query ✓            │
│ h[18]: [-0.03, 0.06, 0.07, ...]  ← "3" query ✓            │
│ h[19]: [0.04, -0.01, 0.02, ...]  ← <p>                     │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┴──────────────────────────┐
            │  NEW: Extract THREE embeddings!       │
            │  1. h[0] - global graph info          │
            │  2. h[17] - query node 0              │
            │  3. h[18] - query node 3              │
            └────────────┬──────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Query Node Extraction                                       │
│                                                             │
│ 1. Find <q> token at position 15                           │
│ 2. Extract query nodes:                                    │
│    - u at position 15 + 2 = 17 (node "0")                  │
│    - v at position 15 + 3 = 18 (node "3")                  │
│                                                             │
│ bos_emb = h[0]  = [0.12, -0.05, 0.08, ..., 0.03]          │
│                   Shape: (16,)                              │
│                                                             │
│ u_emb   = h[17] = [0.05, 0.08, -0.02, ..., 0.01]          │
│                   Shape: (16,)                              │
│                                                             │
│ v_emb   = h[18] = [-0.03, 0.06, 0.07, ..., 0.02]          │
│                   Shape: (16,)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                  Concatenate
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Pooled = concat([bos_emb, u_emb, v_emb])                   │
│                                                             │
│ [0.12, -0.05, ..., 0.03, 0.05, 0.08, ..., 0.01, -0.03, ... │
│  └──── bos_emb ─────┘  └──── u_emb ─────┘  └─── v_emb ───  │
│                                                             │
│ Shape: (48,) = 3 * 16 ← 3x more information!               │
└────────────────────────┬────────────────────────────────────┘
                         │
                  Layer Norm
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Classifier: Linear(48 → 7)                                  │
│                                                             │
│ [1.2, -0.3, 2.5, 0.1, -0.8, 0.2, -1.1]                     │
│  len1  len2  len3  len4  len5  len6  len7                  │
│                  ↑                                          │
│              Predicts len3 ✓                                │
└─────────────────────────────────────────────────────────────┘
```

**Benefit**: 48-dimensional vector with explicit query information!

---

## Side-by-Side Comparison

| Aspect | Before | After |
|--------|--------|-------|
| **Pooling** | h[0] only | h[0] + h[query_u] + h[query_v] |
| **Classifier Input** | 16 dimensions | 48 dimensions |
| **Query Info** | Implicit (in h[0]) | Explicit (separate embeddings) |
| **Parameters** | ~X | ~X + 224 |
| **Complexity** | Same | Same (just different pooling) |

---

## Information Content

### Before
```
[16 numbers encoding everything]
 ↓
Must implicitly encode:
- Graph structure
- Which nodes are queried
- Distance between them
```

### After
```
[16 for graph] + [16 for node 0] + [16 for node 3]
 ↓               ↓                  ↓
Graph structure  Source node info  Target node info

Classifier learns:
"When source = X, target = Y, graph = Z → distance = N"
```

---

## Why Different for IBTT vs AGTT

### IBTT (Index-Based Tokenization)
```
Sequence preserves query structure:
... <q> shortest_distance 0 3 <p>
                          ↑ ↑
                    Easy to find positions!

Query extraction:
1. Find <q> at position i
2. u = position i+2
3. v = position i+3
✓ Precise extraction
```

### AGTT (AutoGraph Trail Tokenization)
```
AutoGraph creates random walks (trails):
<sos> 0 reset 1 0 reset 2 1 0 reset ...
      ↑ No explicit query structure!

Query extraction:
1. Can't find query nodes explicitly
2. Fallback: mean pooling over sequence
3. u_emb = mean(h)
4. v_emb = mean(h)
✗ Less precise, but maintains consistency
```

**Expected**: IBTT outperforms AGTT on shortest_path

---

## Summary

The change gives the classifier **explicit access** to:
1. **Global context**: What the overall graph looks like
2. **Source node**: Characteristics of the query source
3. **Target node**: Characteristics of the query target

Instead of hoping all this information is compressed into a single 16-dimensional vector!
