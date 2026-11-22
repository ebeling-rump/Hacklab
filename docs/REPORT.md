# Learning to Move the Unknown — research report

**Question.** Can TabPFN let a robot adapt faster and better to the unknown contact dynamics of an
unfamiliar object than strong established alternatives?

**Short answer (from our own simulations, not published benchmarks).**
* *Where the classical model is right* (quasi-static pushing of rigid objects), an analytical
  pusher–slider model with online system identification is the strongest predictor and controller.
  No learner beats it there — including TabPFN.
* *Where the world is heterogeneous* (caster-wheeled objects, variable push speeds), a
  **TabPFN residual on top of the identified physics model ("TabPFN-hybrid") is the most accurate
  predictor** after ~20 pushes, and **TabPFN with cross-object experience ("TabPFN transfer-hybrid")
  is the most accurate predictor at every budget from 1 to 20 pushes**, beating both physics + sysID
  and the same transfer construction with a GP, with paired 95% CIs excluding zero.
* TabPFN's uncertainty is **far better calibrated than a GP's in the 2–10-sample regime**.
* Offline gains **did not automatically translate into better closed-loop control**: in the
  heterogeneous tasks TabPFN-hybrid needed *more* pushes than physics + sysID (+3.1, Wilcoxon
  p = 0.015, 24 paired episodes). **With prior experience (transfer) the deficit disappears**
  (+1.25 pushes, p = 0.79) and TabPFN transfer-hybrid places objects most precisely — but it still
  does not beat physics + sysID on pushes. In the standard suite TabPFN-hybrid and physics + sysID are
  statistically indistinguishable (all model-based methods 100% success; −0.5 pushes, p = 0.17).
* Unitree A1 doorway (3D, whole-body pushing, 3 crates): TabPFN transfer-hybrid 3/3, physics + sysID
  2/3 — a compelling demo, but 3 seeds are not statistical evidence.
* TabPFN's CPU inference latency (seconds per planning step on a laptop) is the main practical cost.

Everything below is reproducible with the commands in [`PUSHING.md`](../PUSHING.md).

---

## 1. What was built

| component | summary |
|---|---|
| **Simulator** (`pushlab/sim.py`) | MuJoCo 4. Flat rigid object on a free joint (so the normal-force distribution, and therefore friction torque, follows the hidden CoM), cylindrical finger on two velocity-servoed slide joints with force limits, elliptic friction cones, sensor noise (2 mm, 0.5°), contact-placement noise (2 mm), speed noise (5%). |
| **Objects** (`shapes.py`, `objects.py`) | 8 families: rect, square, disc, L, T, U, plus, Z. Hidden: mass 0.3–2.5 kg, floor friction 0.2–0.7, pusher friction 0.1–0.8, CoM offset up to 35–50% of the half extents, inertia ×0.7–1.4, actuator gain 0.8–1.15. *Heterogeneous regime:* 50% of objects ride on 4 caster wheels (rolling along one hidden axis, sliding across it) and the robot may push at 0.05–0.45 m/s (objects slide/roll after release). |
| **Actions / features** (`actions.py`) | push = (perimeter contact point s, angle φ to the inward normal, distance d, speed v). Models see only observable body-frame geometry: contact point, normal, push direction, φ, d, v, lever arm (+ observable shape descriptor for transfer). Targets: body-frame (Δx, Δy, Δθ). Hidden parameters never enter a model. |
| **Planner** (`planner.py`) | identical for every model: SE(2) lattice (x, y, θ) cost-to-go via Dijkstra around walls (with a recovery gradient out of contact), ~96 candidate pushes predicted per step, depth-2 lookahead by composing body-frame predictions (no extra model calls), uncertainty penalty λ·σ, pusher-reachability constraint, optional decaying exploration bonus (active/dual control). |
| **Challenges** (`tasks.py`) | C1 basic push to position; C2 unknown dynamics (offline learning curves); C3 precision pose (2 cm, 8°); C4 asymmetric objects with large CoM offsets; C5 narrow doorway (object longer than the door is wide, must be rotated); C6 cluttered room with L-route, gap and pillars; C7 transfer to unseen objects/families; `C*H` heterogeneous variants. |
| **3D** (`a1push/`) | Unitree A1 (MuJoCo Menagerie) with an open-loop trot (extended from the repository's `a1_navigation.py` to walk backwards/turn in place), A* navigation around crate and walls, whole-body push primitive (walk to a stance behind the contact point, push with the body, back off), stand-up recovery after falls (counted). Doorway: 0.7 × 0.3 m crate, 0.52 m door, hidden mass 3–6 kg, μ 0.25–0.5, CoM offset. |

### Methods compared (all through the same planner unless noted)

| id | method |
|---|---|
| A `physics_sysid` | quasi-static pusher–slider model (ellipsoidal limit surface + motion cone), MAP estimate of CoM (2), limit-surface radius, pusher friction, actuator gain; Laplace posterior → predictive uncertainty |
| `physics_nominal` | the same model with prior (geometry-only) parameters, no learning |
| B `reactive` | adaptive engineering controller: pushes toward the (carrot) target, lever arm chosen to fix orientation, online RLS of a linear rotation law incl. CoM terms and translation gain; no planner |
| C `gp`, `gp_res`, `gp_hybrid` | per-output ARD-RBF GP (Bauza & Rodriguez-style) on raw targets / on residuals of nominal physics / on residuals of physics + sysID |
| D `tabpfn`, `tabpfn_res`, `tabpfn_physfeat`, `tabpfn_hybrid` | TabPFN v2 (local CPU, 2 ensemble members) on raw targets / residuals / with physics predictions as features / residuals of physics + sysID |
| E `*_active` | same model, planner with a decaying uncertainty bonus; offline: uncertainty sampling vs random experiments |
| F `tabpfn_transfer_hybrid`, `gp_transfer_hybrid` | cross-object context: pushes of 40 previously seen objects (with observable shape descriptor and an `is_target` flag, residuals w.r.t. each object's own physics + sysID) plus the new object's pushes |

---

## 2. Few-shot prediction (Challenge 2 and 7)

Protocol: for each of 24 held-out test objects (all 8 families; T, plus, Z never seen in training
data), models get the first *n* random pushes and predict 40 held-out pushes. Metrics: mean position
/ orientation error of the body-frame outcome, Gaussian NLL, 1σ coverage. Paired comparisons use a
combined score `pos_mm/10 + ang_deg/5` and bootstrap 95% CIs over objects.

### 2.1 Standard regime (fixed speed, rigid sliding objects)
Mean errors (24 objects) — `results/pred_std_*.csv`, figure `results/figures/headline_standard.png`:

| n pushes | physics nominal | **physics + sysID** | GP | GP-hybrid | TabPFN | TabPFN-hybrid |
|---|---|---|---|---|---|---|
| 0 | 13.0 mm / 7.9° | 13.0 / 7.9 | 41.7 / 13.9 | 13.0 / 7.9 | – | 13.0 / 7.9 |
| 3 | 13.0 / 7.9 | **8.5 / 4.6** | 44.8 / 16.6 | 10.5 / 4.9 | 49.5 / 16.9 | 10.6 / 4.8 |
| 10 | 13.0 / 7.9 | **6.5 / 3.0** | 23.6 / 9.0 | 7.8 / 3.2 | 20.5 / 8.0 | 7.1 / 3.1 |
| 40 | 13.0 / 7.9 | 5.6 / 2.4 | 9.7 / 3.8 | 6.0 / 2.5 | 10.7 / 3.7 | **5.5 / 2.4** |

Area-under-learning-curve (n = 1…20, `results/auc_standard.csv`): physics + sysID best; TabPFN-hybrid
second (+0.035, CI [0.002, 0.069]); GP-hybrid third (+0.18). Pure learners need ~40 pushes to approach
the *nominal* physics model. **Physics structure dominates when it is correct.**

### 2.2 Heterogeneous regime (casters + variable push speed)
`results/pred_hetero.csv`, `results/figures/headline_hetero.png`:

| n | physics + sysID | GP-hybrid | TabPFN-hybrid |
|---|---|---|---|
| 3 | 24.2 mm / 8.5° | 25.7 / 8.6 | 24.5 / 8.7 |
| 10 | 21.6 / 6.8 | 19.6 / 7.1 | **19.2 / 6.7** |
| 40 | 20.4 / 6.0 | 13.0 / 4.8 | **12.1 / 5.0** |

Paired TabPFN-hybrid − physics + sysID (score): n=10 −0.27 [−0.89, 0.12]; n=20 −0.62 [−1.32, −0.12];
n=40 −1.03 [−1.88, −0.36], winning on 88% of objects. vs GP-hybrid: −0.05 … −0.12, winning on 54–79%
of objects, CIs include 0. **Calibration** (median NLL; mean NLL at n=3: GP-hybrid 598, TabPFN-hybrid
13, physics + sysID −5.7): GPs with 2–5 points fit near-zero noise and become wildly over-confident;
TabPFN does not need hyperparameter fitting and stays usable.

### 2.3 Transfer to unseen objects (Challenge 7)
Context = pushes of 40 training objects (5 families) in the heterogeneous regime; test on 24 new
objects including 3 unseen families. `results/pred_transfer_het.csv`, `results/figures/transfer_hetero.png`.

| n | physics + sysID | TabPFN-hybrid (no prior) | GP transfer-hybrid | **TabPFN transfer-hybrid** |
|---|---|---|---|---|
| 0 | 4.78 | 4.78 | 5.00 | **4.72** |
| 1 | 4.39 | 4.39 | 4.70 | **4.29** |
| 3 | 4.11 | 4.19 | 4.47 | **3.95** |
| 10 | 3.53 | 3.26 | 3.78 | **3.16** |
| 20 | 3.42 | **2.79** | 3.25 | 2.81 |

(combined score, lower is better). Paired differences of TabPFN transfer-hybrid:
* vs physics + sysID: n=1 −0.10 [−0.15, −0.04]; n=3 −0.17 [−0.30, −0.07]; n=10 −0.37 [−0.77, −0.12];
  n=20 −0.60 [−1.20, −0.20]; wins 83–96% of objects.
* vs GP transfer-hybrid (same context, same features, same physics base): every n from 0 to 20,
  e.g. n=3 −0.53 [−0.76, −0.34], wins 83–100%. **TabPFN uses heterogeneous cross-object context far
  better than a GP** (GP with an ARD kernel over 300+ rows from 40 different objects underfits the
  target object).
* vs TabPFN-hybrid without prior experience: n=1 −0.10 [−0.15, −0.04], n=2 −0.46 [−0.67, −0.28];
  converges by n≈10. **Prior experience is worth ~2–3 pushes of own experimentation.**
* Held-out families (T, plus, Z; 9 objects): still better than physics + sysID and GP transfer at
  n = 1, 3, 10, with smaller margins (−0.06 … −0.09 vs sysID).
* Best median NLL from n = 3 on.

### 2.4 Active experimentation (E)
`experiments/exp_active.py`: pool-based, 60 candidate pushes per object, uncertainty sampling
(argmax of the model's own normalised predictive σ) vs random, 12 steps, 16 heterogeneous test objects.
`results/active_htest.csv`, `results/figures/active_hetero.png`. Mean position error [mm]:

| n | GP-hybrid rand / unc | physics + sysID rand / unc | TabPFN-hybrid rand / unc |
|---|---|---|---|
| 3 | 30.2 / **23.6** | 26.6 / 24.7 | 30.2 / **23.5** |
| 4 | 28.9 / **22.5** | 24.4 / 23.6 | 25.4 / **20.9** |
| 8 | 21.7 / **18.1** | **20.8** / 22.4 | 19.6 / **18.1** |
| 12 | 17.9 / 17.0 | **19.8** / 22.2 | 17.2 / 17.7 |

Paired score difference (uncertainty − random): TabPFN-hybrid −0.25 … −0.43 for n = 2–8 (CI excludes
0 at n = 8: −0.36 [−0.77, −0.02]); GP-hybrid similar (−0.6 [−1.44, 0.01] at n = 4); for physics + sysID
uncertainty sampling is **worse** (+0.45 [0.04, 1.03] at n = 6): its parametric (Laplace) uncertainty
points at pushes that are informative about its *parameters* but not about its *model error*.
**Active experimentation helps the learned residual models by ~2 pushes' worth in the 3–8 push range;
the advantage vanishes by n = 12.** TabPFN and GP benefit similarly.

---

## 3. Closed-loop manipulation (Challenges 1, 3–6)

8 seeds per challenge, identical tasks, identical planner and candidate budget (96), max pushes
15/20/20/40/60. `results/manip_v2.csv`, `results/manip_stats_v2.csv`, `results/figures/manipulation_v2.png`.

| challenge | reactive | physics + sysID | GP-hybrid | TabPFN-hybrid |
|---|---|---|---|---|
| C1 basic | 100%, 6.3 pushes | 100%, 4.9 | 100%, 4.9 | 100%, **4.8** |
| C3 precision pose | 100%, 11.4 | 100%, **8.5** | 100%, 8.9 | 100%, 9.0 |
| C4 asymmetric | 88%, 12.5 | 100%, 7.4 | 100%, 7.3 | 100%, **7.0** |
| C5 doorway | 88%, 20.8, 9.1 wall hits | 100%, 12.1, 1.6 | 100%, 11.9, **0.25** | 100%, **11.4**, 0.5 |
| C6 cluttered | 75%, 46.4, 11.5 | 100%, 38.3, 3.4 | 100%, 42.6, 8.9 | 100%, **36.5**, **2.6** |

Pooled paired (40 episodes) vs physics + sysID: TabPFN-hybrid −0.5 pushes (p = 0.17), −0.38 wall
contacts (p = 0.07); reactive +5.2 pushes (p < 0.001), +3.1 contacts (p = 0.002); GP-hybrid +0.9
pushes (p = 0.07), significantly worse in clutter (+4.4 pushes, p = 0.023).
**All model-based planners solve every task; the reactive baseline fails 10% and hits walls often.
TabPFN-hybrid ≈ physics + sysID.**

Heterogeneous variants (`results/manip_hetero.csv`, 8 seeds):

| challenge | reactive | physics + sysID | GP-hybrid | TabPFN-hybrid |
|---|---|---|---|---|
| C3H precision | 62% | **100%**, 10.3 | 88%, 12.0 | 75%, 12.5 |
| C4H asymmetric | 88% | **100%**, 7.0 | 100%, 8.0 | 100%, 9.6 |
| C5H doorway | 75% | 88%, **14.9** | 88%, 18.8 | 88%, 19.4 |

Pooled: TabPFN-hybrid +3.1 pushes vs physics + sysID (p = 0.015). Both TabPFN failures in C3H were
caster objects; one ended at 15.9 mm / 9.1° (tolerance 20 mm / 8°) at the push limit.

**Why does better offline accuracy not give better control?** The offline test set consists of random
pushes; the planner instead *optimises* over the model and is drawn to actions whose predicted outcome
is optimistic — errors in a flexible residual learned from 3–10 points are exactly such optimistic
spikes (the "optimizer's curse"/model exploitation known from model-based RL). The analytical model is
smooth and its errors are systematic, which a receding-horizon loop corrects. Fast pushes (allowed in
the hetero tasks) magnify this: the residual must extrapolate the coasting distance. The transfer
model, whose residual is anchored by hundreds of rows, is the natural fix; its closed-loop run is in
§6. A larger uncertainty penalty λ for learned residuals is the other obvious fix (not tuned here to
avoid tuning on test seeds).

The active variant (exploration bonus) behaved identically to plain TabPFN-hybrid in the standard
suite: the hybrid model's uncertainty was constant until 2 pushes are observed, so the bonus carried no
signal. After the fix (physics Laplace uncertainty used until the learner has data) it was re-run on the
heterogeneous suite (`results/manip_hetero_active.csv`, same 24 seeds):

| challenge | physics + sysID | TabPFN-hybrid | **TabPFN-hybrid + active** |
|---|---|---|---|
| C3H precision | 100%, 10.3 pushes | 75%, 12.5 | **100%, 9.1** |
| C4H asymmetric | 100%, 7.0 | 100%, 9.6 | 100%, 8.1 |
| C5H doorway | 88%, 14.9 | 88%, 19.4 | 88%, 19.0 |

Pooled: success 96% (= physics + sysID, vs 88% without exploration); −1.75 pushes vs TabPFN-hybrid
(Wilcoxon p = 0.12), +1.4 vs physics + sysID (p = 0.31). **Deliberately informative early pushes
recover most of the closed-loop deficit of the learned residual** — consistent with the offline
active-learning result (§2.4) — but the effect is not significant with 8 seeds per task.

---

## 4. Unitree A1 whole-body pushing (Challenge 8, 3D)

**Physical feasibility.** With the trot gait the A1 pushes 3–6 kg crates with its body and stays
upright in the vast majority of pushes (1 fall in 115 random pushes; 2 falls in each doorway episode,
recovered by a stand-up reset and counted).

**3D few-shot prediction** (`results/a1_prediction.csv`, leave-one-crate-out over 6 crates, 115 pushes):
outcomes of walking pushes are dominated by locomotion variability. Position error stays ≈110–150 mm
for every model over the first 12 pushes — **no model beats the nominal physics model on position**.
Orientation error improves from 18.8° (nominal) to ~11–12° after 12 pushes (physics + sysID 11.2°,
TabPFN transfer-hybrid 11.4°, TabPFN-hybrid 12.1°, GP-hybrid 13.1°). The transfer models are the only
calibrated ones at 1–5 pushes (median NLL ≈ 0 vs 10²–10³ for single-crate GP/TabPFN).

**Doorway demo** (`a1push/episode.py`, videos in `results/videos/A1_*.mp4`). Three crates (seeds;
hidden mass 3–6 kg, μ 0.25–0.5, CoM offset), same start, same planner, budget 30 pushes.

*Development history (reported for transparency):* a first version of the planner (wall margin 4 cm)
gave 1/3 successes for **both** models; both failed by jamming the crate diagonally in the door frame,
a planner flaw (wall contacts are not modelled). The margin was widened to 10 cm for both models and
all seeds were re-run; the table below is the final run.

| seed | physics + sysID | TabPFN transfer-hybrid (prior: 6 other crates, 115 pushes) |
|---|---|---|
| 0 | success, 23 pushes, 1 fall | success, 28 pushes, 2 falls |
| 1 | success, 27 pushes | success, 20 pushes |
| 2 | **failed** — crate jammed in the door frame, then 15 attempts without a walkable path | success, 16 pushes |
| total | 2/3 | **3/3** |

*Follow-up (after the overnight run):* two new crates (seeds 3, 4), same planner and 30-push budget,
comparing physics + sysID with the **API** transfer-hybrid (TabPFN v3.5 via the cloud, same 6-crate
prior; §6b). Note this is a different TabPFN model than seeds 0–2.

| seed | physics + sysID | TabPFN transfer-hybrid (API, v3.5) |
|---|---|---|
| 3 | success, 25 pushes, 2 falls, 7 cm from goal | success, **13 pushes**, 0 falls, 7 cm |
| 4 | success, 26 pushes, 1 fall, 10 cm | success, **11 pushes**, 2 falls, 13 cm |

Videos: `results/videos/A1_doorway_seed3_tabpfn_api_vs_physics.mp4`, `..._seed4_...` (API top,
physics bottom). Over all 5 crates physics + sysID succeeds 4/5 and TabPFN transfer (local or API)
5/5. On the two new crates the API model needs about half the pushes, but the API model also
places the crate slightly less precisely on seed 4.

*Extra demo scenarios* (one seed each, chosen in advance for wheeled complex objects, all results
reported; physics + sysID vs TabPFN v3.5 API transfer-hybrid, planar prior = 40 training objects):

| scenario | physics + sysID | TabPFN API transfer-hybrid | video |
|---|---|---|---|
| A1, L-shaped crate (0.7 × 0.3 m) through the door | success, 15 pushes, 12 cm | success, **8 pushes**, 12 cm | `A1_doorway_Lcrate_tabpfn_api_vs_physics.mp4` |
| C4H: Z-shape on casters to a precise pose | success, **5 pushes**, 20 mm / 5.6° | success, 7 pushes, **11 mm** / 7.0° | `C4H_s1_...mp4` |
| C5H: T-shape on casters through a doorway | success, **13 pushes**, 2 wall contacts | success, 14 pushes, 6 wall contacts | `C5H_s2_...mp4` |
| C6H: L-shape on casters through a cluttered room | success, 46 pushes, 24 mm, 10 wall contacts | success, 43 pushes, **12 mm**, 18 wall contacts | `C6H_s1_...mp4` |

(The A1 L crate was first defined 0.4 m wide; with the 10 cm planner margin it cannot fit the 0.52 m
door at all, so it was slimmed to 0.3 m before any result was recorded.) Mixed picture: the API
model is faster on the A1 crate and more precise in the planar tasks, but touches walls more often.

The robot rotates the 0.7 m crate to pass the 0.52 m door, walks around and through the door after
it, and parks it within 0.15 m of the goal. **With 3–5 seeds this is a demonstration, not statistical
evidence** of a TabPFN advantage; the offline 3D prediction results above show no position-accuracy
advantage, so the difference may be due to chance.

---

## 5. Computational cost

Laptop CPU (4 cores, no GPU), TabPFN v2 with 2 ensemble members, one model per output dimension.

Uncontended, one process, 4 torch threads, one full planning step = fit on n pushes + predict a
96-push candidate library (`experiments/bench_latency.py`, `results/latency.csv`):

| model | n = 5 | n = 20 |
|---|---|---|
| physics + sysID | 0.15 s | 0.13 s |
| GP-hybrid | 0.41 s | 0.48 s |
| TabPFN-hybrid | 1.9 s | 2.2 s |
| TabPFN transfer-hybrid (≈300-row context) | 13.6 s | 14.7 s |

(The benchmark runs above shared the CPU between 2–4 processes, so their per-step times were 2–4×
higher.) TabPFN latency scales with the number of query
rows (~2.5 ms/row/ensemble member), which is why the planner predicts a fixed library of candidate
pushes once per decision and composes it for lookahead. Closed-loop wall time per episode: physics
4–30 s, TabPFN 30–430 s. On a GPU this would be ≲100 ms.

---

## 6. Closed-loop transfer (Challenge 7 in closed loop)

TabPFN transfer-hybrid (context: 40 training objects × ≤8 pushes, heterogeneous regime) on the same
24 heterogeneous task seeds (`results/manip_hetero_transfer.csv`, `results/manip_stats_hetero.csv`):

| challenge | physics + sysID | TabPFN-hybrid | **TabPFN transfer-hybrid** |
|---|---|---|---|
| C3H precision | 100%, 10.3 pushes, 13.6 mm | 75%, 12.5, 16.4 mm | 88%, 10.1, **12.5 mm** |
| C4H asymmetric | 100%, 7.0, 12.1 mm | 100%, 9.6, 11.0 mm | 100%, 7.0, **7.8 mm** |
| C5H doorway | 88%, 14.9, 22.7 mm, 2.9 wall hits | 88%, 19.4, 19.0 mm, 6.8 | 88%, 18.8, **16.6 mm**, 4.1 |

(success, mean pushes, median final position error). Paired vs physics + sysID (24 episodes):
TabPFN-hybrid +3.1 pushes (p = 0.015); **TabPFN transfer-hybrid +1.25 pushes (p = 0.79), +0.4 wall
contacts (p = 0.44)** — prior experience removes the significant closed-loop deficit of the single-object
residual model and gives the most precise final placements, but does not beat physics + sysID on
number of pushes. Cost: ≈45 s of CPU per decision (300-row context, 2 ensemble members × 3 outputs,
shared CPU) vs 0.17 s for physics + sysID.

---

## 6b. Addendum: TabPFN through the cloud API (server model v3.5)

Added after the overnight run. Same learners, with Prior Labs' hosted TabPFN (`tabpfn-client`,
server default model v3.5, 8 estimators) instead of local v2 weights: `tabpfn_api_hybrid`,
`tabpfn_api_transfer_hybrid` in `pushlab/learners.py`. Heterogeneous regime, the **first 8**
held-out test objects only (time budget), same splits and seeds as §2, so the comparison with the
local models is paired per object. Mean position error (mm), `results/pred_api_means.csv`:

| n own pushes | physics + sysID | TabPFN-hybrid (local v2) | TabPFN-hybrid (API) | transfer-hybrid (local v2) | transfer-hybrid (API) |
|---|---|---|---|---|---|
| 1  | 31.5 | 31.5 | 31.5 | 31.5 | **30.2** |
| 3  | 28.7 | 27.3 | 29.0 | 27.1 | **23.1** |
| 5  | 26.5 | 22.0 | 21.4 | 23.4 | **19.3** |
| 10 | 23.4 | 18.3 | 19.6 | 18.9 | **14.5** |
| 20 | 21.9 | 16.1 | 16.4 | 16.0 | **13.0** |

Paired differences with bootstrap 95% CIs (`results/pred_api_paired.csv`):
* **Transfer-hybrid, API vs local:** better on 8/8 objects at every budget; −1.2 mm [−2.2, −0.5] at
  n = 1, −3.9 [−7.9, −1.5] at n = 3, −4.4 [−9.2, −1.1] at n = 10, −3.0 [−6.5, −1.0] at n = 20.
* **Transfer-hybrid (API) vs physics + sysID:** better on 8/8 objects at every budget, e.g.
  −5.6 mm [−12.5, −1.6] at n = 3 and −8.9 [−20.9, −1.9] at n = 10.
* **Hybrid without transfer, API vs local:** no consistent difference (−0.5 to +1.8 mm; only n = 5
  has a CI excluding 0). The newer model helps where the context is large and heterogeneous
  (300 rows from 40 other objects), not with 3–20 own pushes.
* At n = 1 the non-transfer hybrids fall back to physics + sysID, hence identical values.

Latency per planning step (fit + predict 3 targets, network included, median): ~6 s fit + ~2–3 s
predict for both API models, measured while four A1 simulations shared the CPU. That is faster than
local transfer-hybrid (~14 s) but slower than local TabPFN-hybrid (~2 s). The API allows 60 fit
requests per minute; the learner retries on HTTP 429. Each predict call is metered at 10k tokens
(one call per target per planning step).

Caveat: 8 objects, one seed each — the direction is consistent (8/8) but effect sizes are uncertain.

---

## 6c. TabPFN vs the Gaussian-process benchmark, head to head

The GP (ARD-RBF, Bauza & Rodriguez-style) is the literature's data-efficient pushing model, so every
TabPFN variant has a GP twin with **identical inputs** (same features, same physics residual, same
transfer context). `experiments/gp_vs_tabpfn.py` → `results/gp_vs_tabpfn.csv`,
`results/figures/gp_vs_tabpfn.png` (24 test objects per regime, paired per object, 95% bootstrap CIs).

**Prediction accuracy (position error, TabPFN minus GP):**

| model pair | regime | n = 3 | n = 5 | n = 10 | n = 20 | TabPFN more accurate on |
|---|---|---|---|---|---|---|
| raw learner | standard | +4.7 [−1.4, 10.9] | **−4.2** [−6.7, −1.7] | **−3.1** [−4.6, −1.6] | +0.4 [−0.4, 1.0] | 46–79% of objects |
| hybrid | standard | +0.1 | +0.1 | **−0.7** [−1.0, −0.4] | **−1.0** [−1.4, −0.7] | 54–96% |
| raw learner | heterogeneous | +1.2 | **−4.8** [−7.5, −2.3] | **−7.9** [−12.3, −4.6] | −0.7 | 58–88% |
| hybrid | heterogeneous | **−1.2** [−2.4, −0.2] | −1.1 | −0.3 | +0.1 | 62–75% |
| transfer-hybrid | heterogeneous | **−4.1** [−5.9, −2.6] | **−3.8** [−5.7, −2.3] | **−3.8** [−5.1, −2.7] | **−2.5** [−3.6, −1.3] | 83–96% |

(mm; bold = CI excludes 0. At n = 1 both hybrids fall back to physics + sysID and tie.)

**Calibration** (share of outcomes inside ±1σ, ideal 68%): with 3–10 pushes the GPs cover only
22–38% — they are confidently wrong — while TabPFN covers 36–67%. With transfer context the GP is
slightly under-confident (77%), TabPFN close to ideal (57–67%).

**Closed loop, heterogeneous C3H/C4H/C5H, 8 seeds each** (`results/manip_hetero_gptransfer.csv`,
`results/manip_gp_vs_tabpfn_summary.csv`):

| method | success | pushes | final error (successes, median) | model time / decision |
|---|---|---|---|---|
| physics + sysID | 96% | **10.7** | 14.1 mm | 0.17 s |
| GP-hybrid | 92% | 12.9 | 12.9 mm | 0.02 s |
| TabPFN-hybrid | 88% | 13.8 | 15.4 mm | 8.5 s* |
| GP transfer-hybrid | 92% | 13.2 | **11.1 mm** | 0.02 s |
| TabPFN transfer-hybrid | 92% | 12.0 | 11.3 mm | 46 s* |

\*contended CPU (several benchmarks in parallel); uncontended ~2 s and ~14 s (§5).
Paired push differences: TabPFN transfer − GP transfer −1.25 [−3.0, +0.4] (p = 0.20); TabPFN-hybrid −
GP-hybrid +0.9 [−0.5, +2.5] (p = 0.52). **In closed loop the two learners are statistically
indistinguishable**; the GP is ~1000× cheaper per decision. Physics + sysID still needs the fewest
pushes.

**Demo scenarios with all three models** (one seed each, chosen before running; videos
`results/videos/*_physics_sysid_vs_gp_transfer_hybrid_vs_tabpfn_api_transfer_hybrid.mp4`):

| scenario | physics + sysID | GP transfer-hybrid | TabPFN v3.5 API transfer-hybrid |
|---|---|---|---|
| C4H Z-shape on casters, precise pose | 5 pushes, 20 mm | 14 pushes, **4 mm / 0.1°** | 7 pushes, 11 mm |
| C5H T-shape on casters, doorway | 13 pushes | **11 pushes** | 14 pushes |
| C6H L-shape on casters, cluttered room | 46 pushes, 24 mm | **failed** (60 pushes, 24 wall contacts) | 43 pushes, **12 mm** |
| A1, L-shaped crate through the door | 15 pushes | 13 pushes | **8 pushes** |
| A1, box crate (seed 3) through the door | 25 pushes, 2 falls | 20 pushes, 1 fall | **13 pushes**, 0 falls |

A1 three-way videos: `results/videos/A1_doorway_Lcrate_tabpfn_vs_gp_vs_physics.mp4`,
`results/videos/A1_doorway_seed3_tabpfn_vs_gp_vs_physics.mp4` (top TabPFN, middle GP, bottom physics).
In these 5 single-seed demos TabPFN needs the fewest pushes in 3 and the GP in 1, and the GP fails
once; that is suggestive but, unlike the 24-episode benchmark above, not a statistical result.

**Bottom line.** TabPFN's clear, consistent advantage over the GP is (1) **cross-object transfer** —
more accurate on 83–96% of unseen objects at every budget — and (2) **honest uncertainty with very few
samples**. Without transfer, accuracy is a near-tie, and in closed loop we could not show a
significant difference; there the GP is far cheaper.

---

## 6d. Challenge 8: a floor with ice patches (spatially varying ground)

Inspired by the team's A1 slip-prediction branch (`feature/tabpfn-slip-fall-prediction`). The floor
has **visible ice patches** (one large patch over half the start–goal corridor plus three small ones;
ice friction 0.03–0.08, hidden; floor 0.2–0.7, hidden). The robot can see where the ice is but not how
slippery it is. Half-on-ice objects turn the opposite way to the same push on grip, and fast pushes
slide on. Code: ice slabs in `pushlab/sim.py`, task `c8_ice` in `pushlab/tasks.py`.

**What TabPFN gets** (`pushlab/ground.py`): three extra columns per row — the share of the object's
footprint on ice, and the centroid of the iced part in the object's body frame. "Blind" variants get
the same data without these columns. "Experience" = 60 pushes each from **20 other objects on 20 other
icy floors** (C8 seeds 1000+); the ice features are object-relative, so they carry over.

**Prediction** (`experiments/exp_ice.py`, `experiments/ice_summary.py`; `results/pred_ice*.csv`,
`results/ice_paired.csv`, `results/figures/ice_prediction.png`). Unseen objects on unseen floors,
40 held-out pushes each, paired per object, 95% bootstrap CIs. **10 of the planned 16 objects** had
finished when this section was written; the remaining runs are in progress.

| own pushes | physics + sysID | GP + experience | TabPFN v2 + exp., ice hidden | TabPFN v2 + exp. + ice | TabPFN v3.5 API + exp. + ice |
|---|---|---|---|---|---|
| 3  | 26.6 | 26.1 | 25.5 | 23.5 | **22.3** |
| 10 | 26.1 | 25.8 | 24.6 | 22.8 | **21.0** |
| 40 | 25.1 | 22.1 | 22.6 | 19.5 | **18.4** |

(mean position error, mm)
* **TabPFN API + experience vs physics + sysID:** better on 10/10 objects at every budget,
  −4.3 [−5.4, −3.3] mm at n = 3 to −6.7 [−7.7, −5.6] at n = 40 (−16% to −27%).
* **TabPFN vs GP, same experience and inputs:** −2.6 to −3.5 mm, better on 90–100% of objects, all CIs
  exclude 0.
* **Visible-ice columns:** worth −1.8 to −3.2 mm (CIs exclude 0) — TabPFN uses *where* the ice is.
* **Without experience** (own pushes only), every learner is worse than physics + sysID below ~20
  pushes; TabPFN leads only from ~40 pushes (`results/pred_ice.csv`, 16 objects). Learning how ice
  behaves needs experience from other floors.

**Closed loop** (`results/manip_ice.csv`, 8 seeds, no experience): every method reaches every goal
(100%); physics + sysID 11.3 pushes, GP-hybrid 12.1, TabPFN-hybrid 11.9, TabPFN blind 12.8. No
meaningful difference — the planner re-plans after every push and the free floor forgives errors.
Demo video with experience, one seed: `results/videos/C8_s0_physics_sysid_vs_gp_transfer_hybrid_vs_tabpfn_api_transfer_hybrid.mp4`
(all three succeed in 8 pushes; final error 15 / 13 / 11 mm).

---

## 7. Limitations and threats to validity

* All results are from our own MuJoCo simulation; the physics baseline (quasi-static, ellipsoidal
  limit surface) is the *correct model class* for most of our objects, which favours it in the
  standard regime. Real objects (deformable, non-uniform floors) are likely closer to our
  heterogeneous regime — but we did not test on hardware or on the MIT pushing dataset.
* 8 seeds per closed-loop cell; many differences are not significant. Offline results use 24 objects.
* No per-method hyperparameter tuning of the planner (λ, horizon) — all methods share settings chosen
  on the dev set with physics + sysID; this may disadvantage the learned residuals in closed loop.
* TabPFN v2 open weights only; newer weights (v2.5+, license acceptance required) untested.
* A1 gait is open-loop with a stand-up reset after falls; the 3D demo uses one doorway seed per model.
* The reactive baseline is our own design (no established implementation exists for this action space).

## 8. What we learned

1. **Physics first, learning on the residual.** Pure learners (GP or TabPFN) are useless with < 10
   pushes; every competitive method uses the analytical model as a prior. TabPFN is the best residual
   learner we tried.
2. **TabPFN's real advantages are (a) calibration in the tiny-data regime and (b) using heterogeneous
   prior experience across objects** (in-context transfer with descriptors), where it clearly beats a
   GP given identical inputs.
3. **Prediction accuracy ≠ control performance.** Closing the loop exposes model exploitation; any
   claim about learned dynamics models for manipulation should be tested in closed loop.
4. **Latency matters**: TabPFN on CPU is seconds per decision — fine for slow quasi-static pushing,
   too slow for reactive control without a GPU.
