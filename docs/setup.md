# Setup Environment
For this benchmark repository, we will be using multiple **graph-learning tasks**, multiple **graph-learning dataset**, as well as multiple **graph-learning architectures** (i.e. MPNN, GAT, Tokenized + Transformer). Each of them will have different conda environments as their exist dependencies issues among them. For now, we include:

- Basic Py-Geometric
- Tokenized + Transformer
- Graph-GPS

Always check on the [datahub.ucsd.edu/services/disk-quota-service/](https://datahub.ucsd.edu/services/disk-quota-service/) to see the quota limit we have first before installing anything.

## Basic Py-Geometric
To use most of thebasic py-geometric codes on performing graph tasks, we can use the following environment:

```bash
conda env create
conda activate glearning_180a
```

## Tokenized Transformer
To use the [graph-token](https://github.com/alip67/graph-token) code along with the vanilla transformer that we implemented, we will be using the same environment as mentioned above (`glearning_180a`). However, as the tokenization process enters the folder of graph-token, it will automatically create a small venv for doing some of the tokenization required process.

## Graph GPS
[Graph GPS](https://github.com/rampasek/GraphGPS) requires a lot of dependencies, which may take up to 10GG without importing any data yet. This is why we need to work with GraphGPS conda environment not in the usual conda folder we put them. Instead of relying on the `/home` directory where quota is limited, we will be using the `/DSC180A_FA25_A00` directory where we ahve improved to 50GB of disk quota on DSMLP. Note that we can also use `/scratch` that has high quota, but this will be deleted everytime when we close the pod, so this is not ideal. Use the following command to se where the `/DSC180A_FA25_A00` directory is mounted on first:

```bash
mount | grep DS
```

It is always a good habit to just leave the cache from the installation to `/scratch` to save some space, so do the following:

```bash
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
```

Instantiating the conda environment directly will serve this effect by following instructions on [GraphGPS repository](https://github.com/rampasek/GraphGPS), for convineiency, we copied it over as well:

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```

> Alternatively, if there exist much more space on the home directory, do the following:
> 
> ```bash
> # make the dirs we’ll use
> mkdir -p /scratch/$USER/conda_pkgs
> mkdir -p ~/private/DSC180-GLearning/conda_envs
> 
> # for THIS shell, force caches/envs away from home
> export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
> export CONDA_ENVS_DIRS=/home/$USER/private/DSC180-GLearning/conda_envs
> export PIP_CACHE_DIR=/scratch/$USER/pip_cache
>
> # stop using the broken cache in home & free space
> conda deactivate 2>/dev/null || true
> rm -rf ~/.conda/pkgs   # safe: just a cache of downloads/extracts
> conda clean -a -y      # remove any remaining index cache etc.
> 
> # create/activate env in PERSISTENT workspace (not /scratch)
> source /opt/conda/etc/profile.d/conda.sh
> conda create -p ~/private/DSC180-GLearning/conda_envs/graphgps python=3.10 -y
> conda activate ~/private/DSC180-GLearning/conda_envs/graphgps
> ```

Then the rest will be just following the instructions like the one posted on the [GraphGPS repository](https://github.com/rampasek/GraphGPS).