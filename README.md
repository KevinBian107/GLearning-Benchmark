# DSC180-GLearning
Practice Codebase for DSC180 Capstone Graph Learning & Graph Tokenization. The [documentation](/docs/) folder contains information about models we are training here.

## Environment
Refer to [this documentation](/docs/setup.md) for various different environmental setup.

## Running the Codebase
### Graph Tokenization
Clone [Ali Parviz's graph-token repository](https://github.com/alip67/graph-token) for generating synthetic graphs and  tokenize them by following certain rules. First cd into it:

```python
# auto setup enviornment + make graphs
bash graph_generator.sh

# auto setup enviornment + tokenize for tasks
bash task_generator.sh
```

This should automatically setup the environment for the graph-token repository and generate the training graphs and tasks. Then we need to switch the path name and the `split` argument in the `sh` file to `test` to generate the test directory. Then run the following to train an simple transformer for this task:

```bash
python train_gtt.py \
  --graph_token_root graph-token \
  --task cycle_check \
  --algorithm er \
  --run_name gtt-cycle-check \
  --wandb_project graph-token
```

### Notebooks
- [IMBD & MUTAG GLearning Example](notebooks/simple_data.ipynb)
- [Comparing GCN & GAT Example](notebooks/gcn_gat.ipynb)
