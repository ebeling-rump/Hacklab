# Exercise 1: Vision-Guided Pick-and-Place with a Franka Emika Panda

## Submission Deadline: November 4th, 2026 23:59

**Language:** Python 3 with MuJoCo  

A simulated Franka Emika Panda with a wrist-mounted RGB-D camera and a parallel gripper must find
coloured cubes on a table and put them into a bin, **using only what a real robot would know**:
camera images, joint encoders, and a configured scene (bin and obstacle poses). You build the whole
perception-to-action chain.

## 1. Setup

```bash
cd .\ex1_pick_and_place\          # Go to the exercise directory
python -m venv .venv              # Creates a virtual environment (you may use `python3` on Linux)
# Activate the environment
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/macOS

pip install -r requirements.txt   # Installs the dependencies for the exercise.
```

### Watching the simulation live (debugging)

Every run can open the MuJoCo viewer so you can *see* what your agent does. You need a desktop
session with a display (not a headless container).  
On macOS, MuJoCo's passive viewer needs the `mjpython` launcher (as per the
MuJoCo documentation) instead of `python`.  

```bash
python scripts/run_episode.py scene_configs/three_cubes.yaml --viewer     # orbit camera, real time with display
python scripts/run_episode.py scene_configs/obstacles.yaml --viewer --camera wrist      # see through the robot's camera
python scripts/run_episode.py scene_configs/slots.yaml --viewer --speed 0.25 --contacts # slow motion, draw contact points/forces
python scripts/run_episode.py scene_configs/distractors.yaml      # runs headless
python scripts/run_episode.py --random-seed 3 --cubes 4 --obstacles 1 --randomize-bin --hide-bin  # a random scene with randomized hidden bin
python scripts/run_episode.py configs/single_cube.yaml --show-bin     # reveal the pose to the agent (debugging)
```

`--speed S` = simulated seconds per wall second (1 = real time, 0 = as fast as possible).  
`--camera wrist` = shows the image source of your perception code from the robot's point of view.  
`--contacts` = draws contact points and forces (useful for grasp debugging). Mouse: the usual MuJoCo
viewer controls (drag to orbit/pan, scroll to zoom).  
Closing the window lets the run finish headless. If no window can be opened you get a warning and
the run continues headless.  
Note the viewer shows the *whole* world including cube poses, only use it to understand failures.
Your code must still work from the camera image alone.  

## 2. The world

* **Base frame = world.** Robot at the origin, table surface is the plane z = 0, x forward, z up.
* **Cubes:** 4 cm edge, six possible colours (`red green blue yellow orange purple`), random
  position/yaw on the table.
* **Bin:** open tan box, 24 × 24 cm inside, 8 cm walls, 1 cm wall thickness. Cubes must end up
  inside. In most scenes its **pose is hidden** (`task.bin.xy` and `task.bin.yaw` are `None`) and
  you must localise it from the camera; it lies on either side of the robot, with its centre
  at x 0.30–0.50 m, |y| 0.36–0.46 m and any yaw.
* **Obstacles:** fixed boxes on the table (optional).
* **Camera:** 320 × 240, intrinsics `robot.K`, mounted on the hand, hand-eye calibration
  `robot.T_hand_cam`.
* **Gripper:** parallel jaws, width 0 (closed) … 0.08 m (open). The TCP (tool centre point) lies
  0.1034 m from the hand origin along hand z.
* **What you know in advance** (`robot.task`, a `TaskInfo`): which colours to move (`pick_colors`),
  cube size, the bin's **dimensions** and a rough prior for its centre (`task.bin.search_x`,
  `task.bin.search_y_abs`), obstacle boxes, the time limit; the bin **pose** and requested slots
  only when the scene says so.  
* **What you do not know:** where the cubes are and (in hidden-bin scenes) where the bin is.

### Coordinate frames

All frames right-handed. `T_a_b` = pose of frame b in frame a.

| Frame | Definition |
| ----- | ---------- |
| `base` = `world` | origin at the robot base, x forward, z up, table surface z = 0 |
| `hand` | MuJoCo body `hand`: z from the flange towards the fingertips, fingers open/close along hand y. In the home pose: hand z = −z_base, hand x = +y_base, hand y = +x_base |
| `tcp` | `T_hand_tcp` = translation 0.1034 m along hand z (between the finger pads) |
| `bin` | origin at the bin centre on the table, yaw `bin.yaw` about z; `x, y` along the bin walls |
| canonical top-down grasp | `R_DOWN` = [[0,1,0],[1,0,0],[0,0,−1]] (the home orientation); grasp yaw ψ: `R = R_z(ψ)·R_DOWN`, finger axis (hand y) = (cos ψ, sin ψ, 0) |

### Bin observation poses

The camera looks ≈ 9 cm towards −y of the hand, so a top-down hand at `y = ±0.41 + 0.09` centres the
view on a bin at `y = ±0.41`. The two poses are TCP (0.43, +0.50, 0.55) m and (0.43, −0.32, 0.55) m,
hand orientation `R_DOWN`. For every bin centre in the sampling region (x 0.30–0.50, |y| 0.36–0.46)
and every yaw all eight corners of the bin (floor and wall top) project at least 20 px inside the
image. From the cube observation pose the bin may not be (completely) visible.

## 3. API you may use

`student/agent.py` receives a `RobotInterface`:

| Member | Meaning |
| ------ | ------- |
| `get_observation()` | `Observation(rgb, depth, q, dq, gripper_width, time)`; renders the camera (slow-ish) |
| `get_joint_state()` | `(q, dq, gripper_width)` without rendering |
| `set_joint_targets(q)` | position-servo set-points for the 7 joints |
| `set_gripper_width(w)` | commanded finger gap |
| `step(n=1)` | advance `n` control periods |
| `time` | simulated seconds |
| `K`, `image_size`, `T_hand_cam`, `T_hand_tcp`, `q_home`, `control_dt` | calibration and constants |
| `task` | the `TaskInfo`: `bin`, `bin_floor_top`, `bin_to_base`, `base_to_bin`, `in_bin_xy`, and `task.with_bin_pose(xy, yaw)` which returns a copy with your estimated bin pose. Frame helpers raise `BinPoseUnknown` while the pose is hidden |
| `load_robot_model()` | a **private** kinematic `mujoco.MjModel` of the robot (no scene, no cubes) for Jacobians/IK; body `hand`, joints `joint1..7` |
| `collision_checker(margin=0.015, task=None)` | `in_collision(q)`, `segment_free(qa, qb)` for the arm against table, obstacles and the bin (cubes not included). With a hidden bin the default checker has **no bin**; build a new one with `task=task.with_bin_pose(...)` after localising it |

Allowed libraries: `numpy`, `scipy`, and standard libraries except `os, sys, subprocess, ctypes,
importlib, inspect, gc, socket, pickle, shutil, builtins` (the integrity check rejects these
imports). **Not allowed:** anything that reads simulator ground truth (see §6).

## 4. Tasks

Except T1.1 which resides in `pickplace\`, all the other function below exists in `student/` with
its signature and a `NotImplementedError`. **Do not change signatures or names.**

### T1.1 Calibration: `calibration.T_HAND_TCP` & `calibration.T_HAND_CAM`

Construct the proper TCP to panda hand and panda hand to camera transformation matrix.

### T1.2 Perception: `perception.detect_cubes(rgb, colors) -> list[Detection]`

Find one `Detection(color, mask, centroid_px, area_px)` per blob of each requested colour. Threshold
in HSV, label connected components with `scipy.ndimage.label`.

### T2.1 Cube localisation: `localization.estimate_cube_pose(det, depth, K, T_base_cam, cube_size) -> CubeEstimate | None`

From the mask and depth, return the cube **centre** in the base frame and its yaw reduced to (−π/4,
π/4]. Suggestion: back-project mask pixels and transform to the base frame.

### T2.2 Bin localisation: `localization.estimate_bin_pose(rgb, depth, K, T_base_cam, bin_info) -> BinEstimate | None`

Localise the hidden bin from one observation with the bin in view and return its **centre (x, y)**
and **yaw** (reduced to (−π/4, π/4]; the square bin looks the same after a 90° turn, so that is all
that can be observed) as `BinEstimate(position, yaw, n_points)`. Return `None` if no bin of the
right size is visible. You know `bin_info.inner/outer/wall_height` and the prior `search_x`,
`search_y_abs`. Suggestion: Colour may not be enough here as the bin is tan, close to the grey
table.

After a successful estimate the agent calls `task = task.with_bin_pose(est.position, est.yaw)` and
rebuilds its collision checker. Slots are not available with a hidden bin.

### T3 Transforms: `transforms.py` (ten small functions)

Implement these functions: `make_T`, `invert_T`, `rot_z`, `dh_transform`, `fk_hand`, `camera_pose`,
`transform_points`, `backproject`, `backproject_depth`, `project`.

### T4 Planning: `planning.py`

* `plan_task(estimates, task, occupied_bin_xy=())` → ordered `PickPlaceGoal(cube, place_xy,
  place_yaw)`.
* `solve_ik(model, T_target, q_init)` → Inverse kinematics for body `hand`, must respect joint
  limits, return `None` if not converged.
* `cartesian_path(model, q_start, T_target, step=0.01)` → straight-line hand motion as joint
  waypoints.
* `plan_path(q_start, q_goal, checker, rng)` → collision-free joint-space path (straight segment
  first, else RRT/RRT-Connect + shortcutting); reproducible with the given `rng`.

### T5 Control: `control.py`

`JointTrajectory(waypoints)` with `duration` and `sample(t) -> (q, dq)`: trapezoidal profile along
the piecewise-linear joint path, zero velocity at both ends, passes all waypoints, must respect
`V_MAX`/`A_MAX`. `execute_trajectory(robot, traj, monitor)` plays it in simulated real time and
aborts if `monitor()` returns False.
`settle(robot, q_goal, tol, timeout)` holds a goal until converged.

### T6 Grasping: `grasp.py`

`hand_pose_for_tcp`, `plan_grasp`, `plan_place` (above bin cell, release 8–40 mm above the floor),
`close_gripper` (returns the final finger gap, stops when the fingers stop), `open_gripper`.
Top-down only: hand z = −z_base, finger axis along a cube face normal.

### T7 Recovery: `recovery.py`

`classify_grasp(width, cube_size)` → `HELD / EMPTY / MISALIGNED`, `still_holding(width, cube_size)`
(transport monitor), `RecoveryPolicy.next_action(color, status)`: `CONTINUE`, `RESCAN_RETRY` (up to
`max_attempts` failures per colour), then `GIVE_UP`.

### T8 and T9: the agent methods `locate_bin` and `pick_and_place` (`agent.py`)

`agent.py` holds `PickPlaceAgent`. Provided (read it, refine it if you like, do not rename it):
`__init__`, `ik_candidates`, `goto_pose`, `line_to`, `scan`, `bin_scan_poses` and the main loop
`run` *(find the bin → scan → detect → plan task → pick & place one cube → repeat)*. The two methods
below are **yours**.

1. `locate_bin(self) -> bool`
2. `pick_and_place(self, goal) -> bool`

Provided helpers you can use:

* `goto_pose(T_hand, gripper_width=0.08, monitor=None, require=None)` is a collision-checked
  joint-space move; it tries IK candidates ranked by distance to the nominal posture and skips a
  candidate for which `require(q)` is False. `line_to(T_hand, monitor=None)` is a straight Cartesian
  move. Both return `False` on failure and stop early when `monitor()` returns `False`.
* `bin_scan_poses()` returns the two hand poses from which a bin anywhere in the prior is completely
  visible.

## 5. Scenarios and failure cases the grader uses

Variable cube counts (1–6) and colours, random cube yaw, a bin that is **hidden and placed
randomly** (either side, any yaw) in most scenes, rotated, with requested slots and distractor cubes
that must stay put, 0–2 obstacles, sensor noise, a cube nudged after your first look at the whole
workspace, and a cube that slips out of the gripper during lift. Seeds are fixed; some visible
scenarios are in `scene_configs/`.

## 6. Constraints

1. Perception and planning may use **only** `Observation`, `RobotInterface`, `TaskInfo`. Touching
   the simulator's internals (`robot._env`, `gt_*`, `pickplace.env/grading/scene`, `mujoco` data of
   the main simulation, file/OS/process access) fails the integrity check and **voids the whole
   score**.
2. Do not edit any other files in `pickplace/` other than `calibration.py`. Your submission is only
   the files in `student/` and `pickplace\calibration.py`.
3. Use no cube-position knowledge from the config files; those are only to give some scenes for you
   to work with and the grader will use various random position placement.
