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

**Expected output:**
```
======================================================================
Testing Data Consistency: cycle_check / er / train
======================================================================

Loading graph-native dataset (GraphTokenDataset)...
  ✓ Loaded 1000 graphs

Loading sequence dataset (TokenDataset via load_examples)...
  ✓ Loaded 1000 sequences

-------------------------- Test 1: Sample Count ---------------------------
  ✓ PASS: Both have 1000 samples

------------------------ Test 2: Label Consistency ------------------------
  ✓ PASS: All 1000 labels match

--------------------- Test 3: Graph Structure Consistency ------------------
  ✓ PASS: All checked structures match

----------------------- Test 4: Label Distribution ------------------------
  Graph dataset: 487/1000 positive (48.7%)
  Seq dataset:   487/1000 positive (48.7%)
  ✓ PASS: Same label distribution

======================================================================
  ✓ ALL TESTS PASSED
======================================================================
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

**Expected output:**
```
======================================================================
Test 1: Label Distribution Analysis
======================================================================

TRAIN: 1000 samples | 487 positive (48.7%) | 513 negative (51.3%)
VAL  : 200 samples  | 96 positive (48.0%)  | 104 negative (52.0%)
TEST : 200 samples  | 101 positive (50.5%) | 99 negative (49.5%)

------------------------- Baseline Accuracies --------------------------

TRAIN:
  Random guessing (50/50):        50.0%
  Majority class prediction:      51.3%
  Random (matching distribution): 50.0%

------------------------------ Assessment ------------------------------
  ✓ TRAIN: Balanced (48.7% positive)
  ✓ VAL: Balanced (48.0% positive)
  ✓ TEST: Balanced (50.5% positive)

======================================================================
Test 2: Data Leakage Analysis
======================================================================

Loaded train: 1000 samples
Loaded val: 200 samples
Loaded test: 200 samples

----------------------- Computing graph hashes... ----------------------
  train: 1000 unique graphs (from 1000 total)
  val: 200 unique graphs (from 200 total)
  test: 200 unique graphs (from 200 total)

------------------------ Checking for overlaps... ----------------------
  ✓ TRAIN ∩ VAL: No duplicates
  ✓ TRAIN ∩ TEST: No duplicates
  ✓ VAL ∩ TEST: No duplicates

  ✓ No data leakage detected

======================================================================
Test 3: Task Difficulty Assessment
======================================================================

[Manual verification of samples...]

------------------------ Manual Verification ---------------------------
  Correct: 10/10 (100.0%)
  ✓ Labels are correct (verified with DFS)

--------------------------- Task Difficulty ----------------------------
  7/10 sampled graphs have cycles
  Cycle detection is algorithmically simple (DFS traversal)
  → Neural networks should learn this easily

======================================================================
Test 4: Untrained Model Performance
======================================================================

Testing untrained MPNN model (random weights)...

  Untrained model accuracy: 48.3% (483/1000)

  Prediction distribution: {0: 520, 1: 480}
  True label distribution: {1: 487, 0: 513}

------------------------------ Assessment ------------------------------
  ✓ Untrained model performs close to random (expected)

```

---

### 3. `check_input.py` - Input Format Checker

**Purpose**: Analyze input formats and identify prediction types for all tasks.

```bash
python test/check_input.py
```

---