"""Arm definitions for the planner ablation (wxie/planner-ablation).

Each arm is (planner id, {numeric: value}). Planner ids follow
mjpc/planners/include.cc: 0 Sampling (predictive sampling), 5 Cross Entropy,
7 iCEM (the shipped controller, agent_planner=7), 9 MPPI.

Shipped numerics the arms inherit unless overridden (Lean_H12_Magpie.xml):
  sampling_trajectories 20, sampling_spline_points 3, agent_horizon 1.0 s,
  agent_timestep 0.010, sampling_exploration 0.12, std_min 0.01, n_elite 6,
  explore_fraction 0, icem_alpha 0.7, icem_elite_keep 2, mppi_temperature 0.1.

Noise conventions differ by family and that is the confound the equalized arms
remove: PS/MPPI draw N(0, (exploration x half-ctrlrange)^2) per knot per
actuator (0.12 x 0.6-1.8 rad = 0.07-0.21 rad); CEM/iCEM draw N(0, max(elite
std, std_min)^2) in raw ctrl units, and the elite std collapses to the 0.01
floor within ~1 s (log-variance drifts -0.21 per refit at n_elite=6). So the
shipped iCEM plans at sigma = 0.01 rad and the shipped PS at 7-21x that.
"""

PS, CEM, ICEM, MPPI = 0, 5, 7, 9

ARMS = {
    # ---- tier 1: the four planners at the numerics the model ships ----------
    "icem":  (ICEM, {}),
    "cem":   (CEM,  {}),
    "ps":    (PS,   {}),
    "mppi":  (MPPI, {}),

    # ---- tier 2: PS / MPPI with the noise put on CEM's footing --------------
    # raw = std in ctrl units; zero = zero-order-hold spline like CEM/iCEM
    "ps_raw01_zero":   (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                             "sampling_representation": 0}),
    "ps_raw01_cubic":  (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                             "sampling_representation": 2}),
    "ps_raw03_zero":   (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.03,
                             "sampling_representation": 0}),
    "ps_raw10_zero":   (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.10,
                             "sampling_representation": 0}),
    # the one-knob version a user would try first: turn the shipped std down
    "ps_scaled01_cubic": (PS, {"sampling_exploration": 0.01}),
    # MPPI fold at CEM's noise; lambda ladder spans argmin (0.1, ESS~2 while
    # standing) to the uniform mean (10, ESS~20)
    "mppi_raw01_zero_l0.1": (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                    "sampling_representation": 0, "mppi_temperature": 0.1}),
    "mppi_raw01_zero_l1":   (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                    "sampling_representation": 0, "mppi_temperature": 1.0}),
    "mppi_raw01_zero_l10":  (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                    "sampling_representation": 0, "mppi_temperature": 10.0}),

    # ---- tier 3: iCEM internals, one at a time ------------------------------
    "icem_a0":       (ICEM, {"icem_alpha": 0.0}),           # white noise
    "icem_keep0":    (ICEM, {"icem_elite_keep": 0}),        # no elite memory
    "icem_a0_keep0": (ICEM, {"icem_alpha": 0.0, "icem_elite_keep": 0}),  # == CEM
    "icem_ne2":      (ICEM, {"n_elite": 2}),
    "icem_ne10":     (ICEM, {"n_elite": 10}),
    "icem_ne20":     (ICEM, {"n_elite": 20}),               # mean of everything
    "icem_fixed01":  (ICEM, {"cem_variance_fixed": 0.01}),  # adaptation off, at the floor
    "icem_stdmin03": (ICEM, {"std_min": 0.03}),
    "icem_stdmin10": (ICEM, {"std_min": 0.10}),
    "icem_a095":     (ICEM, {"icem_alpha": 0.95}),

    # ---- tier 4: the bridge from CEM's update to PS's -----------------------
    "cem_fixed01":         (CEM, {"cem_variance_fixed": 0.01}),
    "cem_ne1_fixed01":     (CEM, {"cem_variance_fixed": 0.01, "n_elite": 1}),   # argmin of 20 noisy
    "cem_ne1_fixed01_nom": (CEM, {"cem_variance_fixed": 0.01, "n_elite": 1,
                                  "cem_include_nominal": 1}),                   # == PS raw zero
    "cem_ne6_fixed01_nom": (CEM, {"cem_variance_fixed": 0.01, "cem_include_nominal": 1}),
}

BATCHES = {
    "A": ["icem", "cem", "ps", "mppi"],
    "B": ["ps_raw01_zero", "ps_raw01_cubic", "ps_raw03_zero", "ps_raw10_zero",
          "ps_scaled01_cubic", "mppi_raw01_zero_l0.1", "mppi_raw01_zero_l1",
          "mppi_raw01_zero_l10"],
    "C": ["icem_a0", "icem_keep0", "icem_a0_keep0", "icem_ne2", "icem_ne20",
          "icem_fixed01", "icem_stdmin10"],
    "D": ["cem_fixed01", "cem_ne1_fixed01", "cem_ne1_fixed01_nom", "icem_stdmin03",
          "icem_ne10", "icem_a095", "cem_ne6_fixed01_nom"],
}
