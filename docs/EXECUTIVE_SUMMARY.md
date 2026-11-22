# Executive summary

**Branch:** `claude/fewshot-pushing` (from `main`, which holds the Unitree A1 code). Not merged.
Everything ran locally on a 4-core laptop CPU; TabPFN v2 open weights, no paid API calls.

## What was built
* `pushlab/` — a MuJoCo pushing lab: 8 object families with hidden mass, frictions, centre of
  mass, inertia, actuator gain, optional caster wheels; push primitives; an analytical
  pusher–slider model with online system identification; GP and TabPFN learners (raw, residual,
  physics-feature, hybrid, cross-object transfer); one shared planner (SE(2) cost-to-go around
  walls + 2-step lookahead over predicted pushes, uncertainty penalty, optional exploration bonus);
  an adaptive reactive baseline.
* Challenge suite C1–C7 (+ heterogeneous variants): basic pushing, unknown dynamics, precision pose,
  asymmetric objects, narrow doorway, cluttered room, transfer to unseen objects.
* `a1push/` — Challenge 8: the Unitree A1 pushes crates with its own body (MuJoCo contacts),
  navigates around them, and learns online; doorway demo with videos.
* Benchmarks, statistics, figures, animations, report, pitch.

## Which challenges work
All of them run end to end. Model-based planners solve 100% of C1/C3/C4/C5/C6 (8 seeds each);
heterogeneous variants 88–100%; A1 doorway 2/3 (physics) and 3/3 (TabPFN transfer).

## Methods compared
Physics + online sysID (A), adaptive reactive (B), GP / GP-residual / GP-hybrid (C), TabPFN /
TabPFN-residual / -physics-features / -hybrid (D), active-exploration variants (E), TabPFN and GP
transfer-hybrids (F), nominal physics.

## Where TabPFN wins
* **Transfer to unseen objects**: with experience from 40 other objects, TabPFN transfer-hybrid is the
  most accurate predictor at every budget from 1 to 20 pushes in the heterogeneous regime (paired
  CIs exclude 0 vs physics + sysID at n = 1, 3, 5, 10, 20), and beats a GP given *identical* inputs at
  every budget (wins on 83–100% of objects). Prior experience is worth ~2–3 own pushes.
* **Heterogeneous dynamics** (wheels, fast pushes): TabPFN-hybrid beats physics + sysID from ~20 pushes.
* **Calibration with 2–10 samples**: GPs become wildly over-confident; TabPFN does not.
* **Closed loop with transfer**: most precise final object placements in all heterogeneous challenges.

## Where TabPFN loses / ties
* Rigid quasi-static pushing: physics + sysID is best; TabPFN-hybrid is a close second.
* Pure TabPFN (no physics prior) is useless below ~10 pushes — like any pure learner.
* Closed-loop push counts: never significantly better than physics + sysID; without transfer it is
  significantly worse in heterogeneous tasks (model exploitation by the planner).
* 3D A1: walking-push outcomes are so noisy that no model beats nominal physics on position error.
* Latency: seconds (up to ~45 s with a large transfer context) per decision on CPU vs 0.2 s for physics.

## Most compelling demonstrations
* `results/videos/A1_doorway_*.mp4` — the A1 rotates an unknown 70 cm crate through a 52 cm door.
* `results/videos/C5_s3_reactive_vs_physics_sysid_vs_tabpfn_hybrid.mp4` — three controllers, same
  object and doorway (one representative seed).
* `results/figures/paired_hetero.png` — the transfer result in one plot.

## Reproduce
See [`PUSHING.md`](../PUSHING.md); every number in [`REPORT.md`](REPORT.md) comes from a CSV in `results/`.

## Unfinished
* No hardware or MIT-pushing-dataset validation; simulation only.
* Planner hyperparameters not tuned per model (possible disadvantage for learned residuals).
* Active exploration in closed loop re-implemented after the main run but not re-benchmarked.
* TabPFN v2.5/v3 weights untested (license acceptance needed — the user's decision).
* A1: open-loop gait with stand-up resets; 3 doorway seeds per model; no multi-object rearrangement.
