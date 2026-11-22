# Hacklab

## How to use

```python a1_navigation.py```: Run mujoco simulation with Unitree A1 robot navigating in an environment.

```python a1_lidar.py```: Run mujoco simulation with Unitree A1 robot with LiDAR sensors navigating in an environment.

```python a1_dataset.py```: Simulated datasets (/datasets/env_A.npz & env_B.npz).

#### Notebooks
fit_tabPFN.ipynb: should fit the tabPFN model and do the data wrangling (create a table out of the data).

#### TabPFN terrain study
[tabpfn_terrain_nav/](tabpfn_terrain_nav/): simulated study of a wheeled robot with a lidar that learns how slippery the terrain is at each new site from a few hundred self-labelled cells. It compares TabPFN with tree models, neural nets and geometric baselines, runs closed-loop navigation on the predicted costmaps and includes an explainer video. See its README for results and how to rerun.


Simulation environments:

https://pybullet.org/

https://github.com/google-deepmind/mujoco

TabPFN Demo:
https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb
