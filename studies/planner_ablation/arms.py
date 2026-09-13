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

    # ---- tier 5: budget and lookahead within the working family ------------
    "icem_n8_ne2":    (ICEM, {"sampling_trajectories": 8, "n_elite": 2}),
    "icem_n40_ne12":  (ICEM, {"sampling_trajectories": 40, "n_elite": 12}),
    "icem_h05":       (ICEM, {"agent_horizon": 0.5}),
    "icem_h03":       (ICEM, {"agent_horizon": 0.3}),
    "ps_raw01_zero_n40": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                               "sampling_representation": 0, "sampling_trajectories": 40}),
    # PS at the SAME per-plant-step jitter CEM ends up with: the elite mean of
    # 6 draws at sigma has std sigma/sqrt(6) = 0.004 in the directions the cost
    # does not care about; give PS that sigma directly
    "ps_raw004_zero": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.004,
                            "sampling_representation": 0}),
    "ps_raw02_zero":  (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.02,
                            "sampling_representation": 0}),

    # ---- deploy-rate campaign (2026-09-11 afternoon): ladders on CEM, the
    #      planner the real system runs, at the plan rate it runs at ----------
    "cem_ne1":       (CEM, {"n_elite": 1}),      # argmin of 20 noisy (variance -> floor)
    "cem_ne2":       (CEM, {"n_elite": 2}),
    "cem_ne10":      (CEM, {"n_elite": 10}),
    "cem_ne20":      (CEM, {"n_elite": 20}),     # mean of everything, no selection
    "cem_stdmin02":  (CEM, {"std_min": 0.02}),
    "cem_stdmin03":  (CEM, {"std_min": 0.03}),
    "cem_stdmin05":  (CEM, {"std_min": 0.05}),
    "cem_stdmin10":  (CEM, {"std_min": 0.10}),
    "ps_raw02_cubic": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.02,
                            "sampling_representation": 2}),
    "ps_raw03_cubic": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.03,
                            "sampling_representation": 2}),
    "ps_raw05_cubic": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.05,
                            "sampling_representation": 2}),

    # ---- deploy-gains basin study (2026-09-11 evening): the common sigma axis
    #      for the three update rules, plus N at 33 Hz. Every run in this study
    #      uses lean_bench --gains deploy (the robot's KP/KV on plant + planner).
    "cem_stdmin005":   (CEM, {"std_min": 0.005}),
    "ps_raw005_cubic": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.005,
                             "sampling_representation": 2}),
    "mppi_raw005_zero_l1": (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.005,
                                   "sampling_representation": 0, "mppi_temperature": 1.0}),
    "mppi_raw02_zero_l1":  (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.02,
                                   "sampling_representation": 0, "mppi_temperature": 1.0}),
    "mppi_raw03_zero_l1":  (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.03,
                                   "sampling_representation": 0, "mppi_temperature": 1.0}),
    "mppi_raw05_zero_l1":  (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.05,
                                   "sampling_representation": 0, "mppi_temperature": 1.0}),
    # N axis (elite fraction held at 0.3 for CEM)
    "cem_n8_ne2":    (CEM, {"sampling_trajectories": 8, "n_elite": 2}),
    "cem_n40_ne12":  (CEM, {"sampling_trajectories": 40, "n_elite": 12}),
    "ps_raw01_cubic_n8":  (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 2, "sampling_trajectories": 8}),
    "ps_raw01_cubic_n40": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 2, "sampling_trajectories": 40}),
    "mppi_raw01_zero_l1_n8":  (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                      "sampling_representation": 0, "mppi_temperature": 1.0,
                                      "sampling_trajectories": 8}),
    "mppi_raw01_zero_l1_n40": (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                      "sampling_representation": 0, "mppi_temperature": 1.0,
                                      "sampling_trajectories": 40}),

    # ---- representation x update rule (2026-09-11 15:20): on the deploy plant
    #      at 33 Hz PS at 0.01 rad completes 6/6 with a zero-order hold and 0/6
    #      with the cubic spline it ships with, so the spline is a component.
    #      Complete the 2 x 3 (spline x update rule) at sigma = 0.01.
    "cem_cubic":            (CEM,  {"cem_representation": 2}),
    "icem_cubic":           (ICEM, {"cem_representation": 2}),
    "mppi_raw01_cubic_l1":  (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                    "sampling_representation": 2, "mppi_temperature": 1.0}),
    # the one-knob versions a user would try first: shipped spline, shipped
    # noise convention, exploration turned down to 0.01 (= 0.003-0.03 rad)
    "mppi_scaled01_cubic":  (MPPI, {"sampling_exploration": 0.01}),
    "mppi_raw01_cubic_l0.1": (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                     "sampling_representation": 2, "mppi_temperature": 0.1}),
    # PS sigma axis with the hold spline, so the basin compares update rules
    # at one representation (the cubic PS axis stays as the spline contrast)
    "ps_raw01_zero_n8": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                              "sampling_representation": 0, "sampling_trajectories": 8}),
    "ps_raw005_zero": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.005,
                            "sampling_representation": 0}),
    "ps_raw05_zero":  (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.05,
                            "sampling_representation": 0}),
    # ---- why is iCEM immune to the cubic spline (icem_cubic 6/6 at 33 Hz on
    #      the deploy plant, cem_cubic 1/6, ps cubic 0/6)? Split its two
    #      differences from CEM under the cubic: colored noise vs elite memory.
    "icem_a0_cubic":    (ICEM, {"cem_representation": 2, "icem_alpha": 0.0}),      # white noise, memory
    "icem_keep0_cubic": (ICEM, {"cem_representation": 2, "icem_elite_keep": 0}),   # colored noise, no memory
    # knot-spacing falsifier for the cubic failure: if the transient first knot
    # is the mechanism, denser knots (0.2 s apart instead of 0.5) should rescue
    # the cubic; linear is the intermediate representation
    "ps_raw01_cubic_k6":  (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 2, "sampling_spline_points": 6}),
    "ps_raw01_zero_k6":   (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 0, "sampling_spline_points": 6}),
    "ps_raw01_linear":    (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 1}),
    "cem_cubic_k6":       (CEM, {"cem_representation": 2, "sampling_spline_points": 6}),
    # persistence test (Y result 20:35: colored noise rescues the cubic, 6 knots
    # do not; the hold with 6 knots drops to 2/6): the sampled perturbation of
    # the immediate action has to persist ~0.3 s in the rollout to be selected
    # on. Two knots = 0.5 s hold / a 1 s ramp; alpha 0.95 = longer persistence.
    "ps_raw01_zero_k2":   (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 0, "sampling_spline_points": 2}),
    "ps_raw01_cubic_k2":  (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 2, "sampling_spline_points": 2}),
    "icem_a095_cubic":    (ICEM, {"cem_representation": 2, "icem_alpha": 0.95}),
    "icem_a03_cubic":     (ICEM, {"cem_representation": 2, "icem_alpha": 0.3}),
    # points inside the 0.2-0.3 s band: 4-knot hold (tau_p 0.26), 4-knot cubic
    # (0.20), 5-knot hold (0.20), 4-knot linear (0.22)
    "ps_raw01_zero_k4":   (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 0, "sampling_spline_points": 4}),
    "ps_raw01_cubic_k4":  (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 2, "sampling_spline_points": 4}),
    "ps_raw01_zero_k5":   (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 0, "sampling_spline_points": 5}),
    "ps_raw01_linear_k4": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 1, "sampling_spline_points": 4}),
    # ---- 2026-09-13: why 33 Hz is enough -- horizon, budget and rollout
    #      resolution at the robot's rate (plan-rate floor and latency are
    #      sweep flags, not arms)
    "cem_h03":   (CEM, {"agent_horizon": 0.3}),
    "cem_h05":   (CEM, {"agent_horizon": 0.5}),
    "cem_h20":   (CEM, {"agent_horizon": 2.0}),
    "ps_raw01_zero_h03": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                               "sampling_representation": 0, "agent_horizon": 0.3}),
    "ps_raw01_zero_h05": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                               "sampling_representation": 0, "agent_horizon": 0.5}),
    "cem_n4_ne1": (CEM, {"sampling_trajectories": 4, "n_elite": 1}),
    "ps_raw01_zero_n4": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                              "sampling_representation": 0, "sampling_trajectories": 4}),
    "mppi_raw01_zero_l1_n4": (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                     "sampling_representation": 0, "mppi_temperature": 1.0,
                                     "sampling_trajectories": 4}),
    "cem_dt02":  (CEM, {"agent_timestep": 0.02}),
    # low-rate follow-ups: iCEM's sigma window and CEM's k at sigma 0.02
    "icem_stdmin02": (ICEM, {"std_min": 0.02}),
    "icem_stdmin05": (ICEM, {"std_min": 0.05}),
    "cem_ne2_stdmin02":  (CEM, {"n_elite": 2, "std_min": 0.02}),
    "cem_ne10_stdmin02": (CEM, {"n_elite": 10, "std_min": 0.02}),
    "ps_raw01_zero_dt02": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                "sampling_representation": 0, "agent_timestep": 0.02}),
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
    "E": ["icem_n8_ne2", "icem_n40_ne12", "icem_h05", "icem_h03",
          "ps_raw01_zero_n40", "ps_raw004_zero", "ps_raw02_zero"],
    # deploy-rate campaign: R = the five update rules at every rate; L = the
    # ladders at 33 Hz only
    "R": ["cem", "icem", "mppi_raw01_zero_l1", "ps_raw01_cubic", "ps_raw01_zero"],
    "L": ["cem_ne1", "cem_ne2", "cem_ne10", "cem_ne20",
          "cem_stdmin02", "cem_stdmin03", "cem_stdmin05", "cem_stdmin10",
          "ps_raw02_cubic", "ps_raw03_cubic", "ps_raw05_cubic",
          "mppi_raw01_zero_l0.1", "mppi_raw01_zero_l10"],
    # deploy-gains basin study: S = the common sigma axis (5 x 3 update rules),
    # K = CEM elite count, T = MPPI temperature, N = sample count at 33 Hz
    "S": ["cem_stdmin005", "cem", "cem_stdmin02", "cem_stdmin03", "cem_stdmin05",
          "ps_raw005_cubic", "ps_raw01_cubic", "ps_raw02_cubic", "ps_raw03_cubic",
          "ps_raw05_cubic",
          "mppi_raw005_zero_l1", "mppi_raw01_zero_l1", "mppi_raw02_zero_l1",
          "mppi_raw03_zero_l1", "mppi_raw05_zero_l1"],
    "K": ["cem_ne1", "cem_ne2", "cem_ne10", "cem_ne20"],
    "T": ["mppi_raw01_zero_l0.1", "mppi_raw01_zero_l10"],
    "N": ["cem_n8_ne2", "cem_n40_ne12", "ps_raw01_cubic_n8", "ps_raw01_cubic_n40",
          "ps_raw01_zero_n8", "ps_raw01_zero_n40",
          "mppi_raw01_zero_l1_n8", "mppi_raw01_zero_l1_n40"],
    # X = the spline x update-rule completion plus the one-knob PS/MPPI
    "X": ["cem_cubic", "icem_cubic", "mppi_raw01_cubic_l1", "ps_scaled01_cubic",
          "mppi_scaled01_cubic", "mppi_raw01_cubic_l0.1"],
    # S2 = the PS sigma axis with the hold spline
    "S2": ["ps_raw005_zero", "ps_raw01_zero", "ps_raw02_zero", "ps_raw03_zero", "ps_raw05_zero"],
    # Y = the iCEM-under-cubic split (plus the hold versions of the same two)
    "Y": ["icem_a0_cubic", "icem_keep0_cubic", "icem_a0", "icem_keep0",
          "ps_raw01_cubic_k6", "ps_raw01_zero_k6", "ps_raw01_linear", "cem_cubic_k6"],
    "Z": ["ps_raw01_zero_k2", "ps_raw01_cubic_k2", "icem_a095_cubic", "icem_a03_cubic"],
    "W": ["ps_raw01_zero_k4", "ps_raw01_cubic_k4", "ps_raw01_zero_k5", "ps_raw01_linear_k4"],
    # H = horizon / budget / rollout step at 33 Hz; V = the video arms
    "H": ["cem_h03", "cem_h05", "cem_h20", "ps_raw01_zero_h03", "ps_raw01_zero_h05",
          "cem_n4_ne1", "ps_raw01_zero_n4", "mppi_raw01_zero_l1_n4", "cem_dt02", "ps_raw01_zero_dt02"],
    "V": ["cem", "icem", "ps", "mppi", "ps_raw01_zero", "ps_raw01_cubic", "mppi_raw01_zero_l1"],
    # FS = does a larger step per plan lower the plan-rate floor? (25/16.7/10 Hz)
    "FS": ["cem_stdmin02", "cem_stdmin03", "cem_stdmin05", "ps_raw02_zero", "ps_raw03_zero",
           "ps_raw05_zero", "icem_stdmin03", "mppi_raw02_zero_l1", "mppi_raw03_zero_l1"],
    "FS2": ["icem_stdmin02", "icem_stdmin05", "cem_ne2_stdmin02", "cem_ne10_stdmin02"],
    # M = the four hold rules for the mismatch ladder; MS = the sigma-basin link
    # under mismatch (does a wider sigma buy mismatch tolerance, and for whom?)
    "M": ["cem", "icem", "ps_raw01_zero", "mppi_raw01_zero_l1"],
    "MS": ["cem_stdmin02", "cem_stdmin03", "icem_stdmin03", "ps_raw02_zero", "mppi_raw02_zero_l1"],
}

# ---- 2026-09-13 target grid: strategy 25's 3x3 (target_col_x rows -0.10/0/+0.10 m,
#      target_col_y columns A/B/C = 0/0.12/0.24 m) for the four hold planners at
#      0.01 rad. Generated so the 36 names stay consistent.
_GRID_BASE = {
    "cem": (CEM, {}), "icem": (ICEM, {}),
    "ps_raw01_zero": (PS, {"sampling_noise_raw": 1, "sampling_exploration": 0.01, "sampling_representation": 0}),
    "mppi_raw01_zero_l1": (MPPI, {"sampling_noise_raw": 1, "sampling_exploration": 0.01,
                                  "sampling_representation": 0, "mppi_temperature": 1.0}),
}
BATCHES["G"] = []
for _xi, _x in [("m10", -0.10), ("0", 0.0), ("p10", 0.10)]:
    for _yi, _y in [("A", 0.0), ("B", 0.12), ("C", 0.24)]:
        for _b, (_pl, _nums) in _GRID_BASE.items():
            _name = "%s_tx%s_c%s" % (_b, _xi, _yi)
            ARMS[_name] = (_pl, dict(_nums, target_col_x=_x, target_col_y=_y))
            BATCHES["G"].append(_name)

