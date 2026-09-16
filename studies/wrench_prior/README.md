# wrench_prior — VLM physical-parameter priors for a wrench-controlled push

Started 2026-09-14 as the sim half of resurrecting the 2025 "wrench prompting"
workshop paper (scalingforce.github.io). Page:
`mujoco_mpc/docs/wrench_prior/20260914-wrench_prior_direction.html`.

The question: a VLM's estimate of an object's mass, friction and force limits
is a prior on the PLANNING MODEL of a contact MPC, not a wrench setpoint. How
much does an error in that prior cost, and where does the planner's wrench cap
bind? Measured here on the MJPC `UR5 Magpie` model, whose six controls are a
Cartesian wrench at the TCP (+-10 N, +-1 N.m).

| File | What it is |
|---|---|
| `ur5_push.py` | model construction (plant vs planner from one MjSpec), predictive-sampling MPC over the wrench, the push episode |
| `sweep.py` | sweep A (belief error: mass x0.1..10, mu x0.5/2) and sweep B (wrench cap 0.5..3 x mu m g), 6 objects x 3 seeds, resumable |
| `analyze.py` | tables + figures into `docs/wrench_prior/media/` |
| `e0_probe.py` | the VLM prior query (Gemini), minimal prompt, JSON schema with 90% intervals |
| `runs/sweep1/` | results.jsonl, per-episode npz logs, summary.json |
| `runs/e0_gemini_probe.json` | the 2026-09-14 probe on the two photographed 2025 objects |

Run:

```bash
cd mujoco_mpc/studies/wrench_prior
MUJOCO_GL=egl python3 sweep.py --out runs/sweep1 --jobs 3 --threads 6   # ~35 min
python3 analyze.py --run runs/sweep1
```

Conventions that cost time to learn:

- `mujoco.rollout` resets each data before applying the state and
  `mjSTATE_FULLPHYSICS` has no mocap fields, so the target (a mocap body) has
  to ride in the control array under a `control_spec` that includes
  `mjSTATE_MOCAP_POS`. Otherwise every rollout plans toward the XML's target.
- The push point is defined along the FIXED push direction, not the box->target
  bearing; the bearing flips when the box overshoots and the hand then pushes
  it further away.
- The task XML's global `solref .001 1` makes every first touch a 30-80 N,
  3 ms spike. The object geom gets `solref 0.01 1`, and damage is judged on a
  50 ms low-passed contact force (`peak_f_lp`), not the raw peak.
- Success is 3 cm / 0.5 s hold. Predictive sampling at 64 samples, sigma 0.35
  lands the box at 1-3 cm; the tolerance is at the planner's precision floor,
  so read the continuous columns (`final_err`, `t_done`, `peak_f_lp`) with the
  success counts.
