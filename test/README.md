# Test Suite

This directory contains tests to validate data distribution, consistency and analyze training performance.

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

### 3. `data_distribution_test.py` - Data Distribution Analysis

**Purpose**: Analyze and visualize the distribution of training data across different graph generation algorithms for both cycle_check and shortest_path tasks.

**What it analyzes:**

For **Cycle Check** task:
- ✓ Example graph visualizations from each algorithm
- ✓ Number of cycles distribution (violin plots)
- ✓ Graph size distributions (nodes and edges)
- ✓ Label balance (has cycle vs no cycle)

For **Shortest Path** task:
- ✓ Example graph visualizations from each algorithm
- ✓ Path length distribution by algorithm (line plots)
- ✓ **Combined class distribution across all algorithms** (bar chart with counts)
- ✓ Graph size distributions for unique graphs
- ✓ Summary statistics (samples, avg path length, min/max, avg nodes/edges)

**Algorithms analyzed:**
- ER (Erdős-Rényi): Random graphs
- BA (Barabási-Albert): Scale-free networks
- SBM (Stochastic Block Model): Community structure
- SFN (Scale-Free Network): Power-law degree distribution
- Complete: Fully connected graphs
- Star: Star topology graphs
- Path: Path/line graphs

**Usage:**
```bash
python test/data_distribution_test.py
```

---