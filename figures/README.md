# Publication-Quality Plotting for Graph Learning Benchmark

## Directory Structure

```
figures/
├── figures_data/                    # CSV data files (9 files)
│   ├── cycle_check_acc.csv
│   ├── shortest_path_acc.csv
│   ├── shortest_path_agtt_extra_acc.csv
│   ├── shortest_path_agtt_extra_f1.csv
│   ├── shortest_path_f1.csv
│   ├── shortest_path_loss.csv
│   ├── shortest_path_mpnn_extra_acc.csv
│   ├── shortest_path_mpnn_extra_f1.csv
│   └── zinc_loss.csv
│
├── figures_output/                  # Generated plots (9 PNG files)
│   ├── cycle_check_acc.png
│   ├── shortest_path_acc.png
│   ├── shortest_path_agtt_extra_acc.png
│   ├── shortest_path_agtt_extra_f1.png
│   ├── shortest_path_f1.png
│   ├── shortest_path_loss.png
│   ├── shortest_path_mpnn_extra_acc.png
│   ├── shortest_path_mpnn_extra_f1.png
│   └── zinc_loss.png
│
├── plot_figures.py          # Main plotting script
└── README.md                # This file
```

---

## Quick Start

```bash
cd figures/
python plot_figures.py
```

This will read all CSV files from `data/` and generate publication-quality PNG plots in `output/`.

---

## Improved Features

### 1. Better Color Scheme (Colorblind-Friendly)

**Main Model Comparison:**
- **MPNN** = Strong Blue (#0173B2)
- **GPS/GGPS** = Strong Orange (#DE8F05)
- **IBTT** = Teal/Green (#029E73)
- **AGTT** = Purple/Magenta (#CC78BC)

**Training Dataset Comparisons** (more distinctive colors):

For MPNN trained on different datasets:
- `ba+sbm`: Original Blue (#0173B2)
- `er+sbm+path`: Lighter Blue (#56B4E9)
- `path`: Darker Blue (#004D80)

For AGTT trained on different datasets:
- `ba+sbm`: Original Purple (#CC78BC)
- `path`: Pink-Purple (#E56AAD)
- `er+sbm`: Darker Purple (#9B4F96)

### 2. Clearer Titles

Files with `_extra` in the name now have descriptive titles:
- **Before**: "Shortest Path Mpnn Extra Acc"
- **After**: "Shortest Path - MPNN Training Dataset Comparison (Accuracy)"

This makes it clear that these plots compare the same model trained on different datasets.

### 3. Streamlined Output

- Only PNG files (no PDFs) - one plot per CSV
- All plots in `output/` directory
- All data in `data/` directory
- Clean, organized structure

---

## Plot Details

Each plot includes:
- ✅ **Original curves** (transparent, 15% alpha)
- ✅ **Smoothed curves** (Savitzky-Golay filter, window=11)
- ✅ **Error bands** (from MIN/MAX columns, 12% alpha)
- ✅ **Train vs Val**: Solid lines for training, dashed for validation
- ✅ **Professional styling**: 300 DPI, Times New Roman font, clean grid
- ✅ **Smart legends**: Auto-adjusts columns based on number of entries

---

## Customization

### Adjust Smoothing

Edit `plot_figures.py`, line ~250:

```python
create_plot(csv_file, output_dir, show_original=True, smooth_window=15)
```

Change `smooth_window` from 11 to desired value (7-21 recommended).

### Show Only Validation Curves

```python
# In create_plot function, filter by split
if info['split'] == 'train':
    continue  # Skip training curves
```

### Change Colors

Edit the `MODEL_COLORS` or `DATASET_COLORS` dictionaries at the top of `plot_figures.py`:

```python
MODEL_COLORS = {
    'mpnn': '#YOUR_HEX_COLOR',
    'gps': '#YOUR_HEX_COLOR',
    # ...
}
```

---

## CSV Format

Expected format:

```csv
"Step","model_name (dataset) - train/metric","model_name (dataset) - val/metric"
1,0.5,0.4
2,0.6,0.5
```

**Column naming convention:**
- Format: `{arch}-{model}-{task} ({dataset}) - {split}/{metric}`
- Example: `4l4h32-gps-cycle-check (ba+sbm) - train/acc`

**Optional columns for error bands:**
- `{column_name}__MIN`
- `{column_name}__MAX`

---

## Requirements

```bash
pip install pandas numpy matplotlib scipy seaborn
```

---

## Plot Descriptions

### Main Comparison Plots

1. **`cycle_check_acc.png`**: Cycle detection across all models (MPNN, GPS, IBTT, AGTT)
2. **`shortest_path_acc.png`**: Shortest path prediction accuracy across all models
3. **`shortest_path_f1.png`**: Shortest path F1 scores across all models
4. **`shortest_path_loss.png`**: Shortest path training loss
5. **`zinc_loss.png`**: ZINC molecular property prediction (regression loss)

### Training Dataset Comparison Plots

6. **`shortest_path_mpnn_extra_acc.png`**: MPNN trained on different datasets (accuracy)
7. **`shortest_path_mpnn_extra_f1.png`**: MPNN trained on different datasets (F1)
8. **`shortest_path_agtt_extra_acc.png`**: AGTT trained on different datasets (accuracy)
9. **`shortest_path_agtt_extra_f1.png`**: AGTT trained on different datasets (F1)

---

## Tips for Paper

1. **High-quality figures**: PNG at 300 DPI is publication-ready for most journals
2. **Color consistency**: Same color = same model across all plots
3. **Dataset variations**: Different shades distinguish training dataset variations
4. **LaTeX inclusion**:
   ```latex
   \includegraphics[width=0.8\textwidth]{figures/output/cycle_check_acc.png}
   ```

---

## Troubleshooting

### No plots generated
```bash
# Check if CSVs exist
ls data/*.csv

# Check if output directory was created
ls -la output/
```

### Import errors
```bash
# Install missing packages
pip install pandas numpy matplotlib scipy seaborn
```

### Plotting errors
Check the console output - errors are printed with traceback for debugging.

---

**Version**: 2.0 (Reorganized & Improved)
**Author**: Graph Learning Benchmark Team
**Date**: November 2025
