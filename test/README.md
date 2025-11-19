# Test Suite

This directory contains tests to validate data consistency and analyze training performance.

## Tests

### 1. `graph_seq_test.py` - Data Consistency Test

**Purpose**: Verify that graph-native methods (MPNN, GraphGPS) and sequence-based methods (GTT) receive the same underlying data.

**What it checks:**
- ✓ Same number of samples in both representations
- ✓ Labels match between graph and sequence formats
- ✓ Graph structure (nodes/edges) is consistent
- ✓ Label distribution is identical
- ✓ Samples are in the same order

**Usage:**
```bash
# Test default (cycle_check, er, train split)
python test/graph_seq_test.py

# Test specific configuration
python test/graph_seq_test.py --task cycle_check --algorithm er --split train

# Test validation split
python test/graph_seq_test.py --split val

# Test different task
python test/graph_seq_test.py --task connected_nodes --algorithm ba
```

---

### 2. `train_performance_test.py` - Training Performance Analysis

**Purpose**: Analyze why models start with high accuracy and whether it's expected.

**What it checks:**
- ✓ Label distribution (balanced vs imbalanced)
- ✓ Baseline accuracy (random guessing, majority class)
- ✓ Data leakage (duplicates between train/val/test)
- ✓ Task difficulty (manual verification)
- ✓ Untrained model performance

**Usage:**
```bash
# Run full analysis
python test/train_performance_test.py

# Test specific configuration
python test/train_performance_test.py --task cycle_check --algorithm er

# Check more samples manually
python test/train_performance_test.py --samples 20
```

---

### 3. `data_distribution_test.py` - Data Distribution Test

**Usage:**
```bash
python test/data_distribution_test.py
```

---