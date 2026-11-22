# Pitch (≈4 minutes) — "Learning to Move the Unknown"

Assets: `results/videos/A1_doorway_seed2_tabpfn_transfer_vs_physics.mp4` (A1 + crate + doorway),
`results/videos/C5_s3_reactive_vs_physics_sysid_vs_tabpfn_hybrid.mp4` (planar doorway, three
controllers), `results/figures/transfer_hetero.png`, `results/figures/headline_hetero.png`,
`results/figures/manipulation_v2.png`.

---

**[0:00 – 0:30] The problem** *(A1 video, first push)*
A warehouse robot finds a crate blocking a doorway. Nobody told it what's inside: how heavy, how
slippery, where the weight sits, whether it's on wheels. Push it in the middle and it may swing
sideways and jam in the door. Humans figure this out with two or three nudges. Robots today either
need a hand-tuned model of every object or thousands of trials.

**[0:30 – 1:00] Why it's hard**
Contact is the hardest part of robotics: friction, pressure distribution, slip, wheels. Physics
models are excellent *if* you know the parameters and the object behaves like the model. Learned
models can capture anything, but need lots of data. A robot meeting a new object has neither.

**[1:00 – 1:45] Key insight + how we use TabPFN**
We give the robot three things:
1. a **physics prior** — the classical pusher–slider model, with parameters it identifies online;
2. **TabPFN**, a transformer pre-trained on millions of synthetic prediction problems, that learns
   *what physics gets wrong* from just a handful of pushes, in-context, with no training loop — and
   says how sure it is;
3. **experience** — pushes on other objects it has met before, given to TabPFN as context.
A planner then imagines the outcome of ~100 candidate pushes, chains them two steps ahead, and steers
the object around walls and through doors. After every push, the robot learns.

**[1:45 – 2:45] Results** *(show figures)*
* We built 8 challenges — from basic pushing to asymmetric L/T/Z objects, a doorway narrower than the
  object and a cluttered room — plus a Unitree A1 that pushes crates with its own body in MuJoCo.
* Honest headline #1: for ordinary rigid boxes, classical physics with online system identification
  is already excellent, and TabPFN does not beat it. We report that.
* Headline #2: when the world gets messy — objects on caster wheels, fast pushes that make things
  slide — **TabPFN on top of physics predicts best**, and **with experience from other objects it is
  the most accurate model from the very first push**, on objects of shapes it has never seen.
  Given *exactly the same* information, a Gaussian process — the standard tool for data-efficient
  pushing — is worse at every budget and dangerously over-confident with few samples.
* Headline #3: closing the loop matters. All our model-based planners solve every standard task, while
  a reactive controller fails 10% and keeps hitting walls. In the messy tasks the planner can exploit
  errors of the learned model — the robot still mostly succeeds, but physics alone needs fewer pushes.
  Give TabPFN experience from other objects and that deficit disappears.

**[2:45 – 3:30] The demo** *(A1 video, doorway passage)*
Here's the A1. It has never seen this crate. The blue ghost is what it *expects* to happen; the crate
is where it actually goes. Watch the gap shrink as it learns. It turns the crate — 70 cm long, door
52 cm wide — walks around it, pushes it through, follows it, and parks it. Sixteen pushes.
(Bottom half: the same crate with the physics-only model jams in the door frame. Across 3 crates:
3/3 vs 2/3 — a demo, not a statistic; say so.)

**[3:30 – 4:00] Vision**
Today: pushing. The recipe — physics as prior, a foundation model that adapts in context, and
experience as context — applies to any skill where a robot must learn the consequences of its own
actions from a few tries: opening unknown doors, carrying unknown loads, walking on unknown ground.
Robots that acquire physical skills from experience, not from re-programming.
