🛰 LEMP: Learning-Enabled Motion Planning
-----------------------
**Status: Active Development**

LEMP (Learning-Enabled Motion Planning) is a light-weight framework that combines the power of machine learning with traditional motion planning techniques. With a focus on fast iteration, LEMP provides a rapid and agile solution solution for developing learning algorithms for motion planning tasks.

## Installation

### Create Conda Environment
```bash
$ conda create -n lemp python=3.8
$ conda activate lemp
$ conda install -c conda-forge jupyterlab numpy matplotlib
$ pip install pybullet Pillow scipy
# install torch following the instructions from the pytorch website, for example:
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# install torch-geometric following the instructions from the torch-geometric website, for example:
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
$ pip install torch_geometric
```

### Unzip the Datasets
```bash
cd data
unzip static.zip
unzip dynamic.zip
```

## Quickstart

We provide a bunch of useful notebooks in [examples](./examples).

| Notebook                                   | Description                                                 |
|--------------------------------------------|-------------------------------------------------------------|
| [bit_star_planner.ipynb](examples/bit_star_planner.ipynb)            | Example of the BIT* algorithm for planning.   |
| [dataset.ipynb](examples/dataset.ipynb)                               | Saving and loading dataset for static obstacles.                   |
| [dataset_dynamic.ipynb](examples/dataset_dynamic.ipynb)              | Saving and loading Dataset for dynamic obstacles.|
| [dynamic_gnn_planner.ipynb](examples/dynamic_gnn_planner.ipynb)      | Integration of GNN models with a dynamic planner.           |
| [grouping_robot.ipynb](examples/grouping_robot.ipynb)                | Grouping multiple robot arms as one robot to plan                      |
| [load_environment.ipynb](examples/load_environment.ipynb)            | Visualization of trajectories in environments.              |
| [load_object.ipynb](examples/load_object.ipynb)                      | Load objects / obstacles to the environment.               |
| [load_robot.ipynb](examples/load_robot.ipynb)                        | Load robot to the environment.              |
| [object_follow_trajectory.ipynb](examples/object_follow_trajectory.ipynb) | Trajectory visualization for objects.                      |
| [robot_follow_trajectory.ipynb](examples/robot_follow_trajectory.ipynb) | Trajectory visualization for robots.                      |
| [rrt_star_planner.ipynb](examples/rrt_star_planner.ipynb)              | Example of the RRT* algorithm for planning.                |
| [sipp_planner.ipynb](examples/sipp_planner.ipynb)                      | Example of the SIPP* algorithm for dynamic planning. |
| [static_gnn_planner.ipynb](examples/static_gnn_planner.ipynb)          | Integration of GNN models with a static planner. |
