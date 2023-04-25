LEMP: Learning-Enabled Motion Planning
-----------------------
**Status: Maintenance**

This repo is a collection of research projects for learning-enabled motion planning.

## Installation

### Create Conda Environment
```bash
conda create -n lemp python=3.8
conda activate lemp
conda install -c conda-forge jupyterlab numpy matplotlib
pip install pybullet Pillow scipy
```

### Unzip the Datasets
```bash
cd data
unzip *.zip
```

## Quickstart

We provide a bunch of useful notebooks in [examples](./examples).
