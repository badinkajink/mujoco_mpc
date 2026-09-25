# Cavilon surface-contact prototype

This directory is the first Aim 1 → Aim 2/3 integration slice for Cavilon
application. It converts one provisional monocular Aim 1 mannequin surface and reviewed
applicator trace into a MuJoCo height field, replays a compliant applicator tip,
and records MuJoCo contact force and accumulated coverage.

The prototype deliberately uses a Cartesian end-effector proxy.  It establishes
the task geometry, contact oracle, coverage state, and evaluation outputs before
those quantities are coupled to UR5 kinematics and an MJPC residual.  The branch
is based on the existing UR5-capable `wxie/brace-payload` work.

## Run

From this repository:

```bash
python mjpc/tasks/cavilon/build_contact_replay.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/pid117
```

For headless rendering the script selects EGL by default.  Override
`MUJOCO_GL` if a different MuJoCo rendering backend is required.

## Inputs and outputs

Inputs are the frozen Aim 1 PID117 metric atlas, mannequin mask, and selected
expert-contact path.  Outputs include:

- a metric height-field image and MuJoCo scene;
- a transformed reference path in the local task frame;
- per-frame contact force and coverage measurements;
- close and top-view MP4 replays;
- a JSON summary with source hashes and explicit modeling assumptions.

This is simulation ground truth under an assumed rigid surface and nominal
8 mm circular applicator footprint.  It is not evidence that monocular video
measures physical force or true deposited area.

## Rectangular-pad physical-plausibility sweep

The sphere above is retained only as a plumbing regression.  The follow-up
compares a rigid 20 x 9 x 4 mm pad with a 5 x 3 independently compliant-cell
approximation:

```bash
python mjpc/tasks/cavilon/build_tip_model_sweep.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/tip_model_sweep
```

Those dimensions and the cell mechanics are explicit provisional assumptions.
They must be replaced by applicator metrology, force-displacement, friction,
and deposition measurements before the outputs can support real-to-sim claims.

The controlled orientation sweep aligns pad yaw to the path tangent and varies
tilt relative to the local reconstructed surface normal:

```bash
python mjpc/tasks/cavilon/run_pad_angle_sweep.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/pad_angle_sweep
```

Its 0/15/30/45/60 degree conditions are sensitivity tests, not recovered human
poses. Geometry, compliance, path, and indentation remain fixed. The adopted
controller-facing policy is now the 0-degree flat-facing condition; the sweep
is retained only to bound model-form sensitivity because monocular footage does
not recover credible out-of-plane pad pose.

The labeled flat-pad replay transfers the insertion site into the same tangent
frame and adds simulated contact area, traversed path, live site distance, and
a nursing-video comparison:

```bash
python mjpc/tasks/cavilon/render_flat_pad_augmented.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/pid117/flat_pad_augmented
```

Its force-gated area is oracle state for this reduced-order simulation, not
physical contact or deposited Cavilon.

One kinematic UR5e feasibility render can be reproduced with:

```bash
python mjpc/tasks/cavilon/render_ur5_pad_feasibility.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --ur5e-xml /path/to/mujoco_menagerie/universal_robots_ur5e/ur5e.xml \
  --hfield /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/tip_model_sweep/pid117_mannequin_hfield.png \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/ur5_feasibility
```

This solves one offline IK pose and renders it.  It does not test collision-free
approach, dynamics, contact control, or MJPC optimization.

The trajectory version uses sequential offline IK to make the UR5e follow all
PID117 samples with the pad face aligned to the local surface and fixed yaw:

```bash
python mjpc/tasks/cavilon/render_ur5_trajectory.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --ur5e-xml /path/to/mujoco_menagerie/universal_robots_ur5e/ur5e.xml \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/pid117/ur5_trajectory
```

The sequential trust region prevents display-only IK branch jumps. This is
still a kinematic replay, not a collision-checked trajectory or controlled
robot simulation.

## Whole-arm exclusion planner (September 25 engineering audit)

The legacy trajectory is a failed sterility baseline: 377/535 saved poses
intersect a 25.4 mm-radius insertion-site cylinder (367 involve the arm alone).
Its small pose residual did not establish safety. The corrected offline planner
uses explicit Mink collision pairs, multiple elbow/yaw seeds, and OMPL transfers.
It checks all robot meshes (convex hulls), collision primitives and tool pieces
against the cylinder; it also checks nonadjacent primitive self-collision,
floor clearance and heightfield penetration. A 150 mm offset adapter is an
unmeasured design assumption, not the real applicator. Yaw is free while the
pad normal follows the pad-scale surface. The reference is explicitly projected
away from the exclusion zone; it is not an exact imitation of the source video.

```bash
python -m venv --system-site-packages .venv-cavilon
.venv-cavilon/bin/pip install -r mjpc/tasks/cavilon/requirements-planning.txt
.venv-cavilon/bin/python mjpc/tasks/cavilon/plan_sterile_ur5.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --ur5e-xml /path/to/mujoco_menagerie/universal_robots_ur5e/ur5e.xml \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/sterile_ur5_150mm \
  --offset-mm 150 --no-render
.venv-cavilon/bin/python mjpc/tasks/cavilon/finalize_sterile_ur5.py \
  --cldc-root /home/humanoid/Programs/cldc \
  --output-dir /home/humanoid/Programs/cldc/experiments/aim2_3_simulation_bridge/sterile_ur5_150mm
.venv-cavilon/bin/python -m unittest discover -s mjpc/tasks/cavilon -p 'test_*.py' -v
```

The finalizer measures interpolation tracking error, adds an independent
conservative projected-hull exclusion check, and renders at labeled 4× speed.
Saved poses use 30 Hz planned time. Every retained joint-space segment uses
quintic rest-to-rest timing with 0.6 rad/s and 1.2 rad/s² analytic bounds.
OMPL vertices are preserved; local simplifications are rechecked. Validation
subdivides frame edges to at most 0.002 rad/joint: this is sampled validation,
not a continuous collision certificate. Wall-clock OMPL deadlines can change
which branch succeeds; the saved trajectory and source hashes are the evidence.

Important regressions: Mink filters visual-only planning pairs by default, so
the explicit-pair adapter is required. Its gain is scaled for the displacement
QP. MuJoCo 3.10 can report spurious zero distances for separated long cylinder
pairs; cylindrical robot geoms use conservative projected distances instead.

The audited Cartesian-pad video is a separate physics experiment. Render it
with `render_flat_pad_augmented.py` into `pid117/flat_pad_audited`. It uses actual
cell poses, corrected Y orientation and fractional 33/34-step physics timing.
Loaded-cell XY footprint is an estimator, not exact contact patch or deposition.
Coupling this compliant pad to dynamically controlled UR5 execution is the next
gate, along with a full-body collision model and physical calibration.
