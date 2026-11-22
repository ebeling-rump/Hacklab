# Hacklab

**New: [Learning to Move the Unknown](PUSHING.md)** — few-shot physical learning for robotic pushing with TabPFN (planar lab + Unitree A1). Summary: [docs/EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md).

## How to use

```python a1_navigation.py```: Run mujoco simulation with Unitree A1 robot navigating in an environment.

```python a1_lidar.py```: Run mujoco simulation with Unitree A1 robot with LiDAR sensors navigating in an environment.

```python a1_dataset.py```: Simulated datasets (/datasets/env_A.npz & env_B.npz).

#### Notebooks
fit_tabPFN.ipynb: should fit the tabPFN model and do the data wrangling (create a table out of the data).


Simulation environments:

https://pybullet.org/

https://github.com/google-deepmind/mujoco

TabPFN Demo:
https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb
