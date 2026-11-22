# Pitch (≈4 minutes) — "Learning to Move the Unknown"

Assets (all in `results/videos/` unless noted):
* `A1_doorway_Lcrate_tabpfn_api_vs_physics.mp4` — **opening + demo**: A1, unknown L-shaped crate, doorway.
  TabPFN (top) 8 pushes, physics + sysID (bottom) 15 pushes.
* `A1_doorway_seed3_tabpfn_api_vs_physics.mp4` — same comparison, box crate: 13 vs 25 pushes.
* `A1_doorway_seed2_tabpfn_transfer_vs_physics.mp4` — the one crate physics jammed in the door frame.
* `C6H_s1_physics_sysid_vs_tabpfn_api_transfer_hybrid.mp4` — L-shape on caster wheels in a cluttered room.
* `C5_s3_reactive_vs_physics_sysid_vs_tabpfn_hybrid.mp4` — why planning beats reacting.
* `results/figures/paired_hetero.png`, `results/figures/transfer_hetero.png` — the transfer result.

---

**[0:00 – 0:30] The problem** *(L-crate video, first pushes)*
A warehouse robot finds a crate blocking a doorway. Nobody told it what's inside: how heavy, how
slippery, where the weight sits, whether it's on wheels. Push it in the middle and it swings sideways
and jams in the door. Humans figure this out with two or three nudges. Robots today either need a
hand-tuned model of every object or thousands of trials.

**[0:30 – 1:00] Why it's hard**
Contact is the hardest part of robotics: friction, pressure distribution, slip, wheels. Physics
models are excellent *if* you know the parameters and the object behaves like the model. Learned
models can capture anything, but need lots of data. A robot meeting a new object has neither.

**[1:00 – 1:45] Key insight + how we use TabPFN**
We give the robot three things:
1. a **physics prior** — the classical pusher–slider model, with parameters it identifies online;
2. **TabPFN**, a foundation model for tables, that learns *what physics gets wrong* from a handful of
   pushes, in-context, with no training loop — and says how sure it is;
3. **experience** — pushes on other objects it has met before, given to TabPFN as context.
A planner imagines ~100 candidate pushes, chains them two steps ahead, and steers the object around
walls and through doors. After every push, the robot learns.

**[1:45 – 2:45] Results** *(figures, then cluttered-room clip)*
* 8 challenges — basic pushing to asymmetric L/T/Z objects on caster wheels, a doorway narrower than
  the object, a cluttered room — plus a Unitree A1 that pushes crates with its own body in MuJoCo.
* Honest headline #1: for ordinary rigid boxes, physics with online system identification is already
  excellent; TabPFN does not beat it. We report that.
* Headline #2: when the world gets messy — wheels, fast pushes — **TabPFN on top of physics with
  experience from other objects is the most accurate model from the very first push**, on shapes it
  has never seen. A Gaussian process given *exactly the same* information is worse at every budget
  and dangerously over-confident.
* Headline #3: **the newer TabPFN (v3.5, via Prior Labs' API) widens the gap**: more accurate than our
  local v2 on 8 of 8 unseen objects at every budget, ~9 mm better than physics after 10 pushes
  (−40%). Same code, newer foundation model, better robot.
* Headline #4: closing the loop matters. A reactive controller fails 10% and keeps hitting walls;
  every model-based planner solves the standard tasks. A good model alone isn't enough — the
  planner will exploit its errors — and experience from other objects is what closes that gap.

* Headline #5 (show `results/figures/tabpfn_input_dark.png` first, then the GP slide): every push is
  one row of a table — contact, push, lever arm, optional shape and visible ice → what physics got
  wrong. Against the Gaussian-process benchmark with the *same* rows, TabPFN wins with experience and
  stays honest about uncertainty; in closed loop the two are not significantly different.
* Headline #6 (ice, `results/videos/C8_s0_...mp4`): the ground changes — ice the robot can see but not
  feel. With experience from 20 other icy floors, TabPFN beats physics on 15–16 of 16 new objects and
  floors (~20% less error) and the GP too. Without experience, physics is better early on.

**[2:45 – 3:30] The demo** *(L-crate video, then box-crate seed 3)*
Here's the A1. It has never seen this L-shaped crate. The blue ghost is what it *expects* to happen;
the crate is where it actually goes. It turns the L, walks around it, pushes it through a door
narrower than the crate is long, follows it, and parks it. Top: TabPFN, eight pushes. Bottom:
physics, still working at the door — fifteen pushes.
Across five box crates and the L: TabPFN with experience got all six through; physics five of six,
and needed about twice as many pushes on the three crates where the newer TabPFN was used.
That's a handful of crates — a demo, not a statistic; say so.

**[3:30 – 4:00] Vision**
Today: pushing. The recipe — physics as prior, a foundation model that adapts in context, and
experience as context — applies to any skill where a robot must learn the consequences of its own
actions from a few tries: opening unknown doors, carrying unknown loads, walking on unknown ground.
And it gets better for free as the foundation model improves. Robots that acquire physical skills
from experience, not from re-programming.

---

**If asked:**
* *Is TabPFN always better?* No. Rigid boxes: physics wins. In the planar wheeled demos TabPFN parks
  objects more precisely but touches walls more often, and push counts are similar.
* *Is it fast enough?* ~2 s per decision locally, ~9 s with experience via the API, 0.15 s for physics.
  Fine for a deliberate pushing robot, not for reflexes.
* *Real robot?* Simulation only so far (MuJoCo); no hardware or real-dataset validation yet.
