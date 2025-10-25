# DSC180-GLearning
Practice Codebase for DSC180 Capstone Graph Learning & Graph Tokenization

## Environment
Instantiate in the following:

```
conda env create
conda activate glearning_180a
```

### Setup Debug
If setting on DSMLP runs into not enouh memory issues, try the following by first seeing what is the bottleneck:

```bash
conda deactivate 2>/dev/null || true
du -sh ~/.conda/pkgs ~/.cache/pip ~/.cache/torch ~/.local ~/.conda/envs 2>/dev/null
```

If it is files that remains in `/home/kbian/.conda/pkgs → 8.6 GB`, run the clean script would help:

```bash
conda clean -a -y

# or the clean with rf

rm -rf ~/.conda/pkgs
```

## Graph Tokenization
Clone [Ali Parviz's graph-token repository](https://github.com/alip67/graph-token) for generating synthetic graphs and  tokenize them by following certain rules. First cd into it:

```python
# auto setup enviornment + make graphs
bash graph_generator.sh

# auto setup enviornment + tokenize for tasks
bash task_generator.sh
```

This should automatically setup the environment for the graph-token repository and generate the training graphs and tasks. Then we need to switch the path name and the `split` argument in the `sh` file to `test` to generate the test directory. Then run the following to train an simple transformer for this task:

```bash
python train_tokenized_transformer.py \
  --graph_token_root graph-token \
  --task cycle_check \
  --algorithm er \
  --run_name tt-1 \
  --wandb_project graph-token-cyclecheck
```

## Notebooks
- [IMBD & MUTAG GLearning Example](notebooks/simple_data.ipynb)
- [Comparing GCN & GAT Example](notebooks/gcn_gat.ipynb)
