# Related work and how it shaped this project

Written from the authors' background knowledge of the field (not a fresh systematic search);
citations are to well-known papers. **None of the numbers in this repository are directly
comparable to published numbers**: all our results come from our own MuJoCo simulations
(see "Comparability" at the end).

## Analytical models of planar pushing
* **Mason (1986)**, *Mechanics and planning of manipulator pushing operations* — voting theorem,
  direction of rotation of a pushed object depends on the push line vs. the centre of friction.
* **Goyal, Ruina & Papadopoulos (1991)** — the *limit surface*: the set of friction wrenches a
  sliding object can transmit; the object's twist is normal to it.
* **Lee & Cutkosky (1991); Howe & Cutkosky (1996)** — ellipsoidal approximation of the limit surface.
* **Lynch, Maekawa & Tanie (1992); Lynch & Mason (1996)** — quasi-static pusher-slider model with
  the *motion cone* (sticking vs. sliding contact), stable pushing, planning.
* **Hogan & Rodriguez (2016/2020)** — MPC with mode scheduling for the pusher-slider system.
* **Zhou, Bagnell & Mason (2018)** — convex polynomial limit surfaces, fast stochastic contact model.

→ Our baseline **A (`physics_sysid`)** is exactly this family: ellipsoidal limit surface + motion
cone, integrated over the push, with the unknown centre of mass, limit-surface radius, pusher
friction and actuator gain identified online (MAP + Laplace posterior).

## Data-driven and data-efficient pushing
* **Yu, Bauza, Fazeli & Rodriguez (2016)**, *More than a million ways to be pushed* — the MIT
  MCube planar pushing dataset (11 objects, 4 surfaces, ~10^6 pushes); shows real pushing is
  stochastic and heteroscedastic and that the ellipsoidal model has systematic errors.
* **Bauza & Rodriguez (2017)**, *A probabilistic data-driven model for planar pushing* — GP
  (heteroscedastic) models trained on few hundred pushes outperform the analytical model.
* **Bauza, Hogan & Rodriguez (2018)**, *A data-efficient approach to precise and controlled pushing*
  — GP models with very few samples, used inside MPC for precise pose control.
* **Kloss, Schaal & Bohg (2017/2022)**, *Combining learned and analytical models for predicting
  action effects* — hybrid models generalise better than pure learned ones.
* **Ajay et al. (2018)**, *Augmenting physical simulators with stochastic neural networks* —
  residual learning on top of an analytical pushing simulator.
* **Zeng et al. (2019/2020)**, *TossingBot* — "residual physics" as a general recipe.
* **Agrawal et al. (2016)** *Learning to poke by poking*; **Li et al. (2018)** *Push-Net*;
  **Xu et al. (2019)** *DensePhysNet* — data-hungry deep models (thousands of interactions).
* **Chi et al. (2023)**, *Diffusion Policy* — includes the Push-T planar pushing task, but learns
  from ~hundreds of human demonstrations; not a few-shot system-identification method, so we use
  it only as context.

→ Baseline **C (`gp`, `gp_res`, `gp_hybrid`)** follows Bauza & Rodriguez (ARD-RBF GP per output,
body-frame outputs); `*_res`/`*_hybrid` follow the residual-physics recipe so that GP and TabPFN
get exactly the same physics prior.

## Model-based RL, adaptation, active learning
* **Deisenroth & Rasmussen (2011)** PILCO — GP dynamics for data-efficient control.
* **Chua et al. (2018)** PETS — probabilistic ensembles + CEM/MPC.
* **Nagabandi et al. (2019)** — meta-learned online adaptation of dynamics models.
* **Harrison, Sharma & Pavone (2018)** ALPaCA — meta-learned Bayesian last layer for fast adaptation.
* **MacKay (1992)**, **Houlsby et al. (2011, BALD)** — information-based active learning;
  **Feldbaum (1960s)**, dual control — acting to both reach the goal and learn.
* **Stilman & Kuffner (2005)** — navigation among movable obstacles (rearrangement planning).

→ Our **E (`*_active`)** variants add an uncertainty bonus that decays with experience (a cheap
dual-control approximation); `experiments/exp_active.py` tests uncertainty sampling against random
experiments. Our planner is a sampling MPC over a library of push primitives on top of an SE(2)
lattice cost-to-go (a navigation function), in the spirit of MPC-based pushing.

## TabPFN
* **Hollmann et al. (2023, ICLR)** TabPFN; **Hollmann et al. (2025, Nature)** TabPFN v2 — a
  transformer pre-trained on millions of synthetic tabular tasks that performs Bayesian-style
  in-context prediction: `fit` stores the context, `predict` is one forward pass. Outputs a
  binned predictive distribution (quantiles available), handles missing values and heterogeneous
  features, intended for ≤10k samples.
* Relevance: few-shot regression with calibrated uncertainty and **no hyperparameter fitting** —
  the GP's main weakness with 2–10 points is unstable marginal-likelihood optimisation.
* Constraints observed here: CPU inference cost grows with the number of *query* rows
  (≈2.5 ms per query row per ensemble member on our 4-core laptop CPU), so the planner predicts a
  library of ~64–96 candidate pushes once per decision and composes them for multi-step lookahead.
* We used the open **TabPFN v2** weights. Newer weights (v2.5/v3.x) in `tabpfn` 10.x require
  accepting Prior Labs' license in a browser; we did not do that on the user's behalf.

## Comparability
All experiments are **new simulations** (MuJoCo 4, our own objects and physical-parameter
ranges). We did not reproduce the MCube dataset; error magnitudes (mm/deg) depend on our push
lengths (2–10 cm planar, 8–40 cm for the A1) and are not comparable to numbers in the papers above.
