# Project: TabPFN drives a Franka Panda (hackathon)

Goal: use TabPFN (no gradient training) as a closed-loop policy for pick-and-place in
this MuJoCo simulation, imitating a teacher, and show it adapts to disturbances and
unseen situations.

## Key design decisions

- **Teacher** = a reactive scripted expert in TASK SPACE (not the joint-space planner):
  given TCP pose, cube pose, bin pose and gripper width, it returns the next small TCP
  move (dx, dy, dz, dyaw, clipped) and a gripper command, per phase
  (approach, descend, grasp, lift, carry, lower, release, retreat). It must be callable
  on ANY state, so we can relabel states the student visits (DAgger).
- **Policy state features**: cube position/yaw relative to the TCP (in mm/rad), bin
  centre relative to TCP, gripper width, TCP velocity, phase id. Features, not raw
  pixels.
- **Action rows**: one row per action output (dx, dy, dz, dyaw, grip) with an
  `output_id` feature, so ONE `TabPFNRegressor` predicts all outputs (like the
  "one row per joint" trick).
- **Context retrieval**: per query, take the k≈256 nearest teacher states (KD-tree on
  standardized features) as TabPFN context.
- **Execution**: TCP move -> existing IK (`solve_ik` / Jacobian) -> joint targets -> servo.
  Task-space actions keep the policy robot-agnostic.
- **Learning** = appending rows to the context, never gradient steps.
- Always evaluate on **held-out** cube positions / start poses / seeds.
- Put all new code in a new folder `tabpfn_policy/`. Do not break the existing agent
  (`student/`, `pickplace/`, `scripts/run_episode.py` stay as they are).

## Codebase map

### Layout
- `pickplace/` – simulator + grading (treat as read-only).
  - `env.py` `PickPlaceEnv(cfg, viewer="auto")`: owns MuJoCo model/data. Privileged
    ground truth: `gt_cube_pos(i)`, `gt_cube_speed(i)`, `gt_hand_T()`, `gt_tcp_pos()`,
    `gt_camera_T()`, `gt_robot_contacts()`. `reset()` returns a fresh `RobotInterface`.
    Raises `SimTimeLimit` from `step()` once `cfg.time_limit_s` is exceeded.
  - `interfaces.py` `RobotInterface` (what an agent gets), `Observation`, `TaskInfo`,
    `BinInfo`, joint limits `Q_MIN/Q_MAX`, `V_MAX=1.5 rad/s`, `A_MAX=3 rad/s²`.
  - `config.py` `SceneConfig` (+ `CubeCfg`, `BinCfg`, `FaultCfg`, ...), `load_config(yaml)`,
    `random_scene(seed, n_cubes, ...)`, `randomize_bin(cfg, seed, hide)`. Cubes are sampled
    in x∈[0.30,0.60], y∈[-0.20,0.20]; `Q_HOME`, `CUBE_HALF=0.02`.
  - `evaluator.py` `run_episode(cfg, agent_factory) -> (EpisodeResult, env)`: builds env,
    calls `agent_factory(robot).run()`, grades (placed / collisions / distractors / score).
  - `scene.py` builds the MJCF. **Loads the Panda from `../robots/franka_emika_panda/panda.xml`
    (outside this repo, in `Hacklab/robots/`).**
  - `calibration.py` `T_HAND_TCP` (0.1034 m along hand z), `T_HAND_CAM`, `intrinsics()`,
    `GRIPPER_MAX_WIDTH=0.08`.
  - `collision.py` `CollisionChecker.in_collision(q)`, `.segment_free(qa, qb)` (no cubes).
- `student/` – the existing classical agent (perception → localisation → planning → control).
  - `agent.py` `PickPlaceAgent(robot).run()`: `locate_bin()` → loop {`scan()` → `plan_task()` →
    `pick_and_place(goal)`}. Helpers `goto_pose(T_hand)` (IK + RRT), `line_to(T_hand)`
    (Cartesian), `ik_candidates`, `_top_down_hand_pose(p_tcp, yaw)`. `SCAN_TCP=(0.47,0.09,0.55)`.
  - `transforms.py` `fk_hand(q) -> T_base_hand` (pure DH, no MuJoCo), `camera_pose(q, T_hand_cam)`,
    `make_T`, `invert_T`, `rot_z`, `transform_points`, `backproject_depth`, `project`.
  - `planning.py` `solve_ik(model, T_hand_target, q_init, max_iter=300)` (damped LS on body
    `hand`, clips to joint limits, `None` if not converged), `cartesian_path`, `plan_path`
    (RRT-Connect), `plan_task`, `BIN_GRID`.
  - `grasp.py` `hand_pose_for_tcp(p_tcp, yaw)` (top-down, `R = Rz(yaw) @ R_DOWN`),
    `plan_grasp`, `plan_place`, `close_gripper(robot)` (returns final width),
    `open_gripper(robot, w, duration)`.
  - `perception.py` `detect_cubes(rgb, colors)`; `localization.py` `estimate_cube_pose(...)`
    → `CubeEstimate(color, position, yaw∈(-π/4,π/4], n_points)`, `estimate_bin_pose(...)`.
  - `control.py` `JointTrajectory`, `execute_trajectory`, `settle`. `recovery.py`
    `classify_grasp`, `still_holding`, `RecoveryPolicy`.
- `scripts/run_episode.py` CLI: `python scripts/run_episode.py <scene.yaml> [--viewer]` or
  `--random-seed N --cubes K --obstacles M --randomize-bin --hide-bin/--show-bin`. Must set
  `MUJOCO_GL` before importing mujoco (it does this via `select_gl_backend`).
- `scene_configs/*.yaml` sample scenes, incl. disturbances `fault_nudge.yaml`
  (cube displaced after the first full-workspace camera frame) and `fault_drop.yaml`
  (gripper forced open when the cube is lifted > 8 cm).

### Stepping the simulation
```python
from pickplace.config import load_config, random_scene
from pickplace.env import PickPlaceEnv
cfg = random_scene(seed=3, n_cubes=1)        # or load_config("scene_configs/single_cube.yaml")
env = PickPlaceEnv(cfg, viewer=None)         # viewer=None → headless
robot = env.robot_interface()
robot.step(n)                                # n control periods, control_dt = 0.01 s (5 physics substeps each)
```
Or plug a policy into the grader: any class `Agent(robot)` with `.run()` →
`run_episode(cfg, Agent)`. A 10 Hz policy = set targets, then `robot.step(10)`.

### Reading observations
- `robot.get_joint_state() -> (q[7], dq[7], gripper_width)` — cheap, no rendering; use in
  the control loop. Gripper width is the finger gap (0 closed … 0.08 open).
- `robot.get_observation() -> Observation(rgb, depth, q, dq, gripper_width, time)` — renders
  the 320×240 wrist RGB-D camera (slow; also advances the `nudge_after_obs` fault counter).
- TCP pose: `T_base_tcp = fk_hand(q) @ robot.T_hand_tcp`. For a top-down hand the grasp yaw
  is `ψ = atan2(R[1,1], R[0,1])` (finger axis = hand y = (cos ψ, sin ψ, 0)).
- TCP velocity: finite-difference TCP positions, or the `hand` Jacobian from the private
  model (`robot.load_robot_model()`, `mj_jacBody`) times `dq`.
- Cube pose: from perception (`detect_cubes` + `estimate_cube_pose` with
  `camera_pose(obs.q, robot.T_hand_cam)`), or privileged `env.gt_cube_pos(i)` (only allowed
  outside `student/`, e.g. for the teacher / data generation — be explicit about which one
  the student policy uses).
- Bin pose: `robot.task.bin.xy/yaw` (None when hidden → `estimate_bin_pose`, then
  `task.with_bin_pose(xy, yaw)`), or `env.cfg.bin` (privileged).

### Sending joint targets / gripper
- `robot.set_joint_targets(q[7])` — position-servo set-points, clipped to actuator range.
  The servo tracks them over subsequent `step()` calls; there is no velocity interface.
- `robot.set_gripper_width(w)` — commanded gap, clipped to [0, 0.08]; squeezes ≤ 100 N.
- TCP delta → joint targets: `T_hand = hand_pose_for_tcp(p_tcp + dp, yaw + dyaw)`, then
  `q_tgt = solve_ik(model, T_hand, q_current, max_iter=~50)` (`model = robot.load_robot_model()`).
  Note `solve_ik` allocates a new `MjData` per call and targets the `hand` body, not the TCP.
  Keep per-step moves small (≲1 cm) and joint jumps ≲0.3 rad.

### Gotchas
- `student/` has an import whitelist (no `os, sys, subprocess, ctypes, importlib, inspect,
  gc, socket, pickle, shutil, builtins`) and must not touch `robot._env` / `gt_*`. That rule
  applies to the graded submission, not to `tabpfn_policy/`, but don't import `tabpfn_policy`
  from `student/`.
- `requirements.txt` has only mujoco/numpy/scipy/PyYAML. `tabpfn_policy/` additionally needs
  pandas, pyarrow, tabpfn (10.x), CPU torch, xgboost, matplotlib (installed in `.venv`).
- Evaluator collisions count robot contact with `obstacle*` / `bin_*` geoms per control step.

## tabpfn_policy/ (what exists)
- `expert.py` – reactive task-space teacher. `Expert.act(ExpertState, phase) -> (action, phase)`,
  action = [dx, dy, dz (m), dyaw (rad), grip width (m)]. Phases APPROACH…RETREAT, DONE.
- `runner.py` – `SimEpisode` (TCP move -> `solve_ik` -> joint targets, 20 Hz = 5 sim steps,
  IK target placed `lead=2.5` x the delta ahead because the servo lags), `run_expert_episode`.
- `features.py` – `FEATURES`, `CubeTracker` (camera belief; "in hand" when width says held),
  `perceived_state`, `feature_vector`. Student phase input = `phase_perc`, never true `phase`.
- `collect.py` – parallel collection to `data/teacher.parquet` (+ resumable `data/teacher_parts/`).
- `offline_eval.py` – episode split, KD-tree retrieval, one-row-per-output TabPFNRegressor,
  TabPFNClassifier for grip, XGBoost / k-NN baselines, per-k time budget, plots to `results/`.
- `run_expert.py` – `python -m tabpfn_policy.run_expert --episodes 10 [--noise 0.3]`.

## Lessons (non-obvious, keep)
- The sim gripper squeezes only ~2-4 N: cubes slip on fast moves / wrist turns. The expert therefore
  slows the last 3 cm of descent, holds still while closing, pre-rotates during LIFT, carries slowly
  until yaw-aligned, caps acceleration while holding, and keeps yaw targets within ±(45°+0.1 rad)
  (otherwise joint 7 hits its limit once joint 1 turns to the bin and IK fails forever).
- Set `OPENBLAS_NUM_THREADS=1` in worker processes (collect.py does): OpenBLAS otherwise commits
  ~1 GB per process at numpy/scipy import, which grew the Windows page file to 30 GB.
- TabPFN on this CPU (no GPU): ~1.2 s/query at 256 context rows, ~3.4 s at 1k, ~19 s at 4k rows
  (v2 and v3.5-fast similar). Closed-loop use at 20 Hz needs small k, batching or a GPU.
- TabPFN-3.5 weights need the licence accepted at ux.priorlabs.ai and `TABPFN_TOKEN` (user-level env
  var on this machine). v2 weights need no login. Use `--model-version v3.5-fast`.
- Teacher seeds 0-199 are training data; use seeds >= 10000 for held-out closed-loop evaluation.
