# Learning to Move the Unknown — few-shot physical intelligence for pushing

A robot meets an object it has never seen. It does not know its mass, friction, centre of mass
or how it reacts to contact. It learns from a handful of its own pushes and uses what it learns
to plan — up to rotating a crate through a doorway that is narrower than the crate is long.

* Results, methodology and limitations: [`docs/REPORT.md`](docs/REPORT.md)
* 3–5 minute pitch: [`docs/PITCH.md`](docs/PITCH.md)
* Related work: [`docs/LITERATURE.md`](docs/LITERATURE.md)

## Layout

| path | what |
|---|---|
| `pushlab/shapes.py`, `objects.py` | object footprints (rect, square, disc, L, T, U, plus, Z) + **hidden** physics (mass, floor/pusher friction, CoM, inertia, actuator gain, casters) |
| `pushlab/sim.py` | MuJoCo planar pushing simulator (free-joint object, velocity-servoed finger, sensor/actuation noise) |
| `pushlab/actions.py` | push parameterisation (contact point, push angle, distance, speed) + observable body-frame features |
| `pushlab/physics_model.py` | analytical quasi-static model (ellipsoidal limit surface + motion cone) with online MAP/Laplace system ID |
| `pushlab/learners.py` | GP, TabPFN, residual/hybrid/physics-feature and cross-object transfer wrappers (one interface) |
| `pushlab/planner.py` | shared planner: SE(2) lattice cost-to-go + depth-2 lookahead over predicted push outcomes, uncertainty penalty, active-exploration bonus |
| `pushlab/tasks.py` | challenge suite C1–C6 (+ heterogeneous variants `C*H`) |
| `pushlab/runner.py` | closed-loop episodes + adaptive reactive baseline |
| `a1push/` | Unitree A1 (MuJoCo Menagerie) whole-body pushing: gait, navigation, push primitive, doorway demo, prior-experience data |
| `experiments/` | data generation, prediction / active-learning / manipulation benchmarks, figures, animations |
| `results/` | CSVs, figures (`results/figures`), videos (`results/videos`, `results/a1`) |

## Install

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements-pushing.txt
```

TabPFN runs **locally on CPU** with the open v2 weights (downloaded on first use; no API key).
The A1 model is sparse-cloned from MuJoCo Menagerie into `mujoco_menagerie/` on first use.

## Reproduce

```powershell
# datasets (random pushes on unseen objects; ~1-2 min each)
.venv\Scripts\python experiments/gen_dataset.py --name train  --seed 1  --n-objects 40 --pushes 100 --families train
.venv\Scripts\python experiments/gen_dataset.py --name test   --seed 2  --n-objects 24 --pushes 100 --families all
.venv\Scripts\python experiments/gen_dataset.py --name htrain --seed 11 --n-objects 40 --pushes 100 --families train --regime hetero
.venv\Scripts\python experiments/gen_dataset.py --name htest  --seed 12 --n-objects 24 --pushes 100 --families all   --regime hetero

# few-shot prediction learning curves (Challenge 2), transfer (Challenge 7), active learning
.venv\Scripts\python experiments/exp_prediction.py --data test  --out results/pred_std_fast.csv  --models physics_nominal physics_sysid gp gp_physfeat gp_res gp_hybrid
.venv\Scripts\python experiments/exp_prediction.py --data test  --out results/pred_std_tabpfn.csv --models tabpfn tabpfn_physfeat tabpfn_res tabpfn_hybrid --ns 2 3 5 10 20 40
.venv\Scripts\python experiments/exp_prediction.py --data htest --out results/pred_hetero.csv --models physics_nominal physics_sysid gp gp_physfeat gp_res gp_hybrid tabpfn tabpfn_physfeat tabpfn_res tabpfn_hybrid
.venv\Scripts\python experiments/exp_prediction.py --data htest --source htrain --out results/pred_transfer_het.csv --models tabpfn_transfer_hybrid gp_transfer_hybrid
.venv\Scripts\python experiments/exp_active.py --data htest

# closed-loop challenge suite (same planner for every model)
.venv\Scripts\python experiments/exp_manip.py --challenges C1 C3 C4 C5 C6 --seeds 10 --tag v2 --methods reactive physics_sysid gp_hybrid tabpfn_hybrid tabpfn_hybrid_active
.venv\Scripts\python experiments/exp_manip.py --challenges C3H C4H C5H --seeds 10 --tag hetero --methods reactive physics_sysid gp_hybrid tabpfn_hybrid

# figures + side-by-side animations
.venv\Scripts\python experiments/make_figures.py
.venv\Scripts\python experiments/compare_episode.py --challenge C5 --seed 1 --methods physics_sysid tabpfn_hybrid

# Unitree A1 (3D): prior experience, 3D few-shot prediction, doorway demo with video
.venv\Scripts\python -m a1push.gen_data --crates 6 --pushes 20
.venv\Scripts\python -m a1push.eval_prior
.venv\Scripts\python -m a1push.episode --model tabpfn_hybrid --seed 0 --video
```
