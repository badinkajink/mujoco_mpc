"""Run the AgileRRT of Caldwell & Correll (ISRR 2015) to completion, standalone.

This is the whole algorithm -- tree search included -- against the real MuJoCo
model, deliberately *not* fitted to MJPC's Planner interface. MJPC asks a
planner for an action every control step; AgileRRT answers a different question
("give me one trajectory from start to goal, take as long as you need"), and
forcing it through `ActionFromPolicy` hides what it actually does. Run this way
the tree, its growth rate, and the solution it returns can all be seen.

Structure, following the paper:
  Alg 3/4  steering, in steering.py terms -- reused from steering_test.py
  Alg 5    Extend: nearest neighbour under the steering metric, steer, project,
           collision-check, insert
  Alg 6    the RRT loop over sampled targets and a set of horizons T_h

The output is a solution-trajectory CSV in exactly the format
`corridor_benchmark --dump` emits, so the same filmstrip.py renders it and the
result is directly comparable to the MJPC planners.

Usage:
  python3 agile_rrt_run.py --max-vertices 2000 --out /tmp/agile_rrt.csv
"""
import argparse
import csv
import math
import pathlib
import time

import numpy as np
import mujoco


# ----------------------------------------------------------------- model setup

def default_xml():
    here = pathlib.Path(__file__).resolve()
    # .../mjpc/planners/agile_rrt/prototype/ -> .../mjpc/tasks/...
    return str(here.parents[3] / "tasks" / "triple_pendulum_cartpole" / "task.xml")


EPS, CENTERED = 1e-6, 1


class System:
    """The MuJoCo model plus the few task quantities the planner needs."""

    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_path(xml)
        m = self.model
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.n = 2 * m.nv
        self.dt = m.opt.timestep
        self.u_max = float(m.actuator_ctrlrange[0, 1])

        self.head_site = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
                          for s in ("head1", "head2", "tip")]
        # Discover obstacles by the "obstacle" name prefix rather than by the
        # two names task.xml happens to use, so slalom.xml's six disks are
        # found too. This is the same rule Corridor::Initialize applies in the
        # MJPC task, so the tree and the residual see the same world.
        self.obstacle_geom = [
            g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            .startswith("obstacle")]
        self.obstacle_radius = [m.geom_size[g][0] for g in self.obstacle_geom]

        # Goal and avoidance margin come from the model's <custom> numerics,
        # the same place the MJPC residual reads them, so pointing --xml at a
        # different world moves the goal with it.
        self.goal_x = self.numeric("residual_Goal", 6.0)
        self.clearance_param = self.numeric("residual_Clearance", 0.08)

        # The cart's own joint range is the state-space box to sample over;
        # hard-coding it would silently truncate a longer corridor.
        slider = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slider")
        self.cart_range = tuple(m.jnt_range[slider])
        # scratch mjData reused by the hot paths, so the tree loop does not
        # allocate one per steering evaluation
        self._data = mujoco.MjData(m)

    def numeric(self, name, default):
        """First element of a <custom><numeric>, or `default` if absent."""
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        if i < 0:
            return default
        return float(self.model.numeric_data[self.model.numeric_adr[i]])

    def set_state(self, data, x):
        data.qpos[:] = x[:self.nq]
        data.qvel[:] = x[self.nq:]

    def clearance(self, data):
        """Min over heads/obstacles of (distance - radius), in the x-z plane."""
        out = 1e9
        for h in self.head_site:
            p = data.site_xpos[h]
            for g, r in zip(self.obstacle_geom, self.obstacle_radius):
                c = data.geom_xpos[g]
                d = math.hypot(p[0] - c[0], p[2] - c[2]) - r
                out = min(out, d)
        return out


# ------------------------------------------------- Eq 3, Eq 4: linearization

def rollout_zero(sys, x0, steps):
    """Eq 3: the zero-control trajectory, and Eq 4: (A_k, B_k) along it."""
    m = sys.model
    data = sys._data
    mujoco.mj_resetData(m, data)
    sys.set_state(data, x0)
    mujoco.mj_forward(m, data)

    xs = np.empty((steps + 1, sys.n))
    As = np.empty((steps, sys.n, sys.n))
    Bs = np.empty((steps, sys.n, sys.nu))
    for k in range(steps):
        xs[k] = np.concatenate([data.qpos, data.qvel])
        data.ctrl[:] = 0.0
        mujoco.mjd_transitionFD(m, data, EPS, CENTERED, As[k], Bs[k], None, None)
        mujoco.mj_step(m, data)
    xs[steps] = np.concatenate([data.qpos, data.qvel])
    return xs, As, Bs


# ------------------------------- Eq 21/22, Eq 15, Eq 17: the per-vertex cache

def precompute(sys, As, Bs, Q, R, P1lqr):
    """Alg 6 step 1: K_k (Riccati), then W_K, S_K and the closed-loop STM.

    Done once per vertex at the longest horizon; steering to any target at any
    t_h <= t_h_max reuses the cached slices, which is the paper's central
    efficiency claim.
    """
    steps = len(As)
    n = sys.n
    Rinv = np.linalg.inv(R)

    # Eq 22 backward Riccati -> Eq 21 gains
    P = P1lqr.copy()
    Ks = [None] * steps
    for k in range(steps - 1, -1, -1):
        A, B = As[k], Bs[k]
        S = R + B.T @ P @ B
        Ks[k] = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ Ks[k]
        P = 0.5 * (P + P.T)

    # Eq 15 W_K, Eq 17 S_K, and Phi_K(k,0).
    #
    # These are accumulated in *closed loop* (Acl = A - B K), not open loop.
    # That is not a stylistic choice: the open-loop Gramian of this system has
    # cond ~1e17 at t_h = 1 s, past the point where double precision retains
    # any significant digits, so the open-loop form of Eq 15 cannot be solved
    # at all. The closed-loop reformulation is what makes the method numerable.
    #
    # The 1/dt on B R^-1 B' and the dt on M' R M are not cosmetic. Eq 15 is a
    # continuous-time integral, W(t) = int Phi B_c R^-1 B_c' Phi' ds, but
    # mjd_transitionFD returns the *discrete* B_d ~ B_c dt. Substituting B_d
    # directly makes every W a factor dt too small -- 200x here -- and the
    # steering that comes out of it is correspondingly timid: measured max|u|
    # was 0.21 N against a 20 N limit, so the tree could not move. Dividing by
    # dt restores B_c, and the matching dt on S_K keeps the cost integral
    # consistent with it.
    inv_dt = 1.0 / sys.dt
    W = np.zeros((n, n))
    S = np.zeros((n, n))
    WKs, SKs, Acls = [W.copy()], [S.copy()], []
    for k in range(steps):
        A, B, K = As[k], Bs[k], Ks[k]
        Acl = A - B @ K
        M = K @ W - Rinv @ B.T * inv_dt
        S = Acl @ S @ Acl.T + M.T @ R @ M * sys.dt
        S = 0.5 * (S + S.T)
        W = Acl @ W @ Acl.T + B @ Rinv @ B.T * inv_dt
        W = 0.5 * (W + W.T)
        Acls.append(Acl)
        WKs.append(W.copy())
        SKs.append(S.copy())
    # Only Acl is stored, not the STM Phi_K(k,0).
    #
    # The reference implementation forms Phi_K(k_h,k) as
    # Phi_K(k_h,0) inv(Phi_K(k,0)). That identity is exact but unevaluable
    # here: a stable closed loop makes Acl contractive, so the 200-step product
    # Phi_K(k,0) is numerically singular and the inverse throws outright.
    # steer_trajectory instead accumulates Phi_K(k_h,k) backwards from the
    # identity, which needs no inverse and touches each Acl once.
    return dict(Ks=Ks, Bs=Bs, WKs=WKs, SKs=SKs, Acls=Acls, Rinv=Rinv, R=R,
                inv_dt=inv_dt)


# ------------------------------------------------- Alg 4: inexact linear steering

def steer_cost(pre, xzero, x_des, kh, P1):
    """Alg 4 steps 1,2,5 only: Eq 23 eta*, Eq 25 J. This is the RRT metric.

    Nearest-neighbour calls this once per vertex per horizon, so it is the
    single hottest operation in the planner and is kept free of any trajectory
    construction.
    """
    WK, SK = pre["WKs"][kh], pre["SKs"][kh]
    err = xzero[kh] - x_des
    # lstsq rather than solve/inverse: W_K P1 W_K + S_K stays poorly
    # conditioned even in closed loop, and a hard solve throws where the
    # least-squares solution is still a usable steering direction.
    eta, *_ = np.linalg.lstsq(WK @ P1 @ WK + SK, WK @ P1 @ err, rcond=None)
    resid = err - WK @ eta
    J = 0.5 * eta @ (SK @ eta) + 0.5 * resid @ (P1 @ resid)
    return J, eta


def steer_trajectory(pre, xzero, eta, kh):
    """Alg 4 steps 3,4: Eq 24's z*(t) and v*(t) -- the infeasible reference
    trajectory that the projection then tracks."""
    Ks, Acls, WKs, Rinv, Bs = (pre["Ks"], pre["Acls"], pre["WKs"],
                               pre["Rinv"], pre["Bs"])
    inv_dt = pre["inv_dt"]
    n = WKs[0].shape[0]
    nu = Bs[0].shape[1]
    x_tilde = np.zeros((kh + 1, n))
    u_tilde = np.zeros((kh, nu))

    # Walk k downwards accumulating w_k = Phi_K(kh, k)' eta directly:
    #   Phi_K(kh, k) = Phi_K(kh, k+1) Acl_k  =>  w_k = Acl_k' w_{k+1}
    # starting from Phi_K(kh, kh) = I. No STM is stored and no inverse taken.
    w = eta.copy()
    for k in range(kh, -1, -1):
        x_tilde[k] = xzero[k] - WKs[k] @ w
        if k < kh:
            u_tilde[k] = (Ks[k] @ WKs[k] - Rinv @ Bs[k].T * inv_dt) @ w
        if k > 0:
            w = Acls[k - 1].T @ w
    return x_tilde, u_tilde


# --------------------------------------------- Eq 26: Hauser projection + checks

def project(sys, x0, x_tilde, u_tilde, Ks, kh):
    """Eq 26: roll the true nonlinear dynamics under u = u~ - K(x - x~).

    Feasible by construction, and this is also where collision is decided --
    the obstacles are real MuJoCo geoms, so a contact during the rollout is the
    collision check. No swept line-circle test is needed.
    """
    m = sys.model
    data = mujoco.MjData(m)
    sys.set_state(data, x0)
    mujoco.mj_forward(m, data)

    xs = np.zeros((kh + 1, sys.n))
    us = np.zeros((kh, sys.nu))
    min_clear = 1e9
    collided = False
    for k in range(kh):
        x = np.concatenate([data.qpos, data.qvel])
        xs[k] = x
        u = u_tilde[k] - Ks[k] @ (x - x_tilde[k])
        u = np.clip(u, -sys.u_max, sys.u_max)
        us[k] = u
        data.ctrl[:] = u
        mujoco.mj_step(m, data)
        min_clear = min(min_clear, sys.clearance(data))
        if data.ncon > 0:
            collided = True
            xs[k + 1] = np.concatenate([data.qpos, data.qvel])
            return xs[:k + 2], us[:k + 1], True, min_clear
    xs[kh] = np.concatenate([data.qpos, data.qvel])
    if not np.all(np.isfinite(xs)):
        return xs, us, True, min_clear
    return xs, us, collided, min_clear


# --------------------------------------------------------------- the tree

class Vertex:
    __slots__ = ("x", "parent", "controls", "states", "pre", "xzero", "t")

    def __init__(self, x, parent, controls, states, t):
        self.x = x
        self.parent = parent
        self.controls = controls   # controls that got here from parent
        self.states = states       # states along that edge
        self.pre = None            # lazily-built per-vertex cache
        self.xzero = None
        self.t = t                 # time from root


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class AgileRRT:
    def __init__(self, sys, args):
        self.sys = sys
        self.args = args
        n, nu = sys.n, sys.nu

        self.Q = np.zeros((n, n))
        self.R = np.eye(nu) * args.r_cost
        self.P1 = np.eye(n)
        self.P1lqr = np.eye(n)

        self.horizons = [int(round(t / sys.dt)) for t in args.horizons]
        self.kh_max = max(self.horizons)

        self.x_goal = np.zeros(n)
        self.x_goal[0] = sys.goal_x

        self.vertices = []
        self.rng = np.random.default_rng(args.seed)
        self.stats = dict(extends=0, collisions=0, steer_fail=0, nn_time=0.0,
                          pre_time=0.0, extend_time=0.0)

    # --- the state-space box the planner samples from
    def sample(self):
        if self.rng.random() < self.args.goal_bias:
            return self.x_goal.copy()
        x = np.empty(self.sys.n)
        lo, hi = self.sys.cart_range
        x[0] = self.rng.uniform(lo, hi)                  # cart
        x[1:4] = self.rng.uniform(-np.pi, np.pi, 3)     # link angles
        x[4] = self.rng.uniform(-4.0, 4.0)              # cart velocity
        x[5:8] = self.rng.uniform(-8.0, 8.0, 3)         # link rates
        return x

    def dist_to_goal(self, x):
        d = x - self.x_goal
        d[1:4] = wrap(d[1:4])
        # position and angle errors dominate; velocities are weighted down so
        # the goal check is not dominated by a fast-moving near-miss
        return math.sqrt(np.sum(d[:4] ** 2) + 0.25 * np.sum(d[4:] ** 2))

    def ensure_cache(self, v):
        """Build the vertex's linearization + Gramian cache on first use."""
        if v.pre is not None:
            return
        t0 = time.perf_counter()
        xzero, As, Bs = rollout_zero(self.sys, v.x, self.kh_max)
        v.xzero = xzero
        v.pre = precompute(self.sys, As, Bs, self.Q, self.R, self.P1lqr)
        self.stats["pre_time"] += time.perf_counter() - t0

    def nearest(self, x_rand):
        """Alg 5 step 1: nearest under the Eq-25 steering cost, over all
        vertices and all horizons in T_h."""
        t0 = time.perf_counter()
        best = (np.inf, None, None, None)
        for v in self.vertices:
            self.ensure_cache(v)
            for kh in self.horizons:
                try:
                    J, eta = steer_cost(v.pre, v.xzero, x_rand, kh, self.P1)
                except np.linalg.LinAlgError:
                    continue
                if np.isfinite(J) and J < best[0]:
                    best = (J, v, kh, eta)
        self.stats["nn_time"] += time.perf_counter() - t0
        return best

    def extend(self, x_rand):
        J, v, kh, eta = self.nearest(x_rand)
        if v is None:
            self.stats["steer_fail"] += 1
            return None
        t0 = time.perf_counter()
        try:
            x_tilde, u_tilde = steer_trajectory(v.pre, v.xzero, eta, kh)
        except np.linalg.LinAlgError:
            self.stats["steer_fail"] += 1
            return None
        xs, us, collided, _ = project(self.sys, v.x, x_tilde, u_tilde,
                                      v.pre["Ks"], kh)
        self.stats["extend_time"] += time.perf_counter() - t0
        self.stats["extends"] += 1
        if collided or len(us) == 0:
            self.stats["collisions"] += 1
            return None
        child = Vertex(xs[-1], v, us, xs, v.t + len(us) * self.sys.dt)
        self.vertices.append(child)
        return child

    def solve(self):
        root = Vertex(np.zeros(self.sys.n), None, np.zeros((0, self.sys.nu)),
                      None, 0.0)
        self.vertices.append(root)

        best_d = self.dist_to_goal(root.x)
        best_v = root
        t_start = time.perf_counter()
        it = 0
        while (len(self.vertices) < self.args.max_vertices and
               time.perf_counter() - t_start < self.args.time_limit):
            it += 1
            child = self.extend(self.sample())
            if child is not None:
                d = self.dist_to_goal(child.x)
                if d < best_d:
                    best_d, best_v = d, child
                    if self.args.verbose:
                        print(f"  [{len(self.vertices):5d} verts, "
                              f"{time.perf_counter()-t_start:6.1f}s] "
                              f"closest to goal: {d:.3f}  "
                              f"(cart {child.x[0]:+.2f}, t={child.t:.2f}s)")
                if d < self.args.goal_tolerance:
                    print(f"\nGOAL REACHED after {len(self.vertices)} vertices, "
                          f"{time.perf_counter()-t_start:.1f}s")
                    return best_v, best_d, time.perf_counter() - t_start, True
        return best_v, best_d, time.perf_counter() - t_start, False


# ------------------------------------------------------ solution -> dump CSV

def extract_path(v):
    """Walk parent pointers back to the root and concatenate the edges."""
    chain = []
    while v.parent is not None:
        chain.append(v)
        v = v.parent
    chain.reverse()
    us = [e.controls for e in chain]
    return (np.concatenate(us) if us else np.zeros((0, 1)))


def replay_and_dump(sys, controls, path):
    """Replay the solution controls through the model and write the CSV that
    filmstrip.py reads, including the same cost terms the MJPC task uses so the
    two are directly comparable."""
    m = sys.model
    data = mujoco.MjData(m)
    mujoco.mj_resetData(m, data)
    mujoco.mj_forward(m, data)

    # weights from task.xml's <sensor> user data: [norm, weight, lo, hi]
    W = {name: m.sensor_user[
             mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)][1]
         for name in ("Cart", "Upright", "Velocity", "Control", "Avoidance")}
    clearance_param = sys.clearance_param

    rows = []
    for k in range(len(controls) + 1):
        u = float(controls[k][0]) if k < len(controls) else 0.0
        data.ctrl[0] = u
        mujoco.mj_forward(m, data)

        cost = W["Cart"] * abs(data.qpos[0] - sys.goal_x)
        cost += W["Upright"] * sum(abs(math.cos(data.qpos[1 + i]) - 1.0)
                                   for i in range(3))
        cost += W["Velocity"] * float(np.sum(np.abs(data.qvel)))
        cost += W["Control"] * abs(u)
        avoid = 0.0
        for h in sys.head_site:
            p = data.site_xpos[h]
            for g, r in zip(sys.obstacle_geom, sys.obstacle_radius):
                c = data.geom_xpos[g]
                d = math.hypot(p[0] - c[0], p[2] - c[2])
                avoid += max(0.0, r + clearance_param - d)
        cost += W["Avoidance"] * avoid

        rows.append(dict(
            step=k, time=k * sys.dt,
            cart=data.qpos[0], th1=data.qpos[1], th2=data.qpos[2],
            th3=data.qpos[3], dcart=data.qvel[0], dth1=data.qvel[1],
            dth2=data.qvel[2], dth3=data.qvel[3],
            ctrl=u, cost=cost, ncon=data.ncon,
            min_clearance=sys.clearance(data)))
        if k < len(controls):
            mujoco.mj_step(m, data)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default=default_xml())
    p.add_argument("--out", default="/tmp/agile_rrt.csv")
    p.add_argument("--max-vertices", type=int, default=1500)
    p.add_argument("--time-limit", type=float, default=900.0)
    p.add_argument("--horizons", type=float, nargs="+",
                   default=[0.2, 0.5, 1.0],
                   help="T_h, the set of steering horizons in seconds")
    p.add_argument("--goal-bias", type=float, default=0.1)
    p.add_argument("--goal-tolerance", type=float, default=0.5)
    p.add_argument("--r-cost", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()

    sys_ = System(args.xml)
    print(f"model: nq={sys_.nq} nv={sys_.nv} nu={sys_.nu} dt={sys_.dt} "
          f"|u|<={sys_.u_max}")
    print(f"horizons T_h = {args.horizons} s "
          f"({[int(round(t/sys_.dt)) for t in args.horizons]} steps)")
    print(f"growing tree to at most {args.max_vertices} vertices "
          f"(time limit {args.time_limit:.0f}s)\n")

    rrt = AgileRRT(sys_, args)
    best_v, best_d, elapsed, solved = rrt.solve()

    s = rrt.stats
    print(f"\ntree: {len(rrt.vertices)} vertices in {elapsed:.1f}s")
    print(f"  extends attempted   : {s['extends']}")
    print(f"  rejected (collision): {s['collisions']} "
          f"({100.0*s['collisions']/max(1,s['extends']):.1f}%)")
    print(f"  steering failures   : {s['steer_fail']}")
    print(f"  time in nearest-nbr : {s['nn_time']:.1f}s "
          f"({100.0*s['nn_time']/max(1e-9,elapsed):.0f}%)")
    print(f"  time in per-vertex precompute : {s['pre_time']:.1f}s "
          f"({100.0*s['pre_time']/max(1e-9,elapsed):.0f}%)")
    print(f"  time in extend (steer+project): {s['extend_time']:.1f}s "
          f"({100.0*s['extend_time']/max(1e-9,elapsed):.0f}%)")
    print(f"  best distance to goal: {best_d:.3f} "
          f"({'SOLVED' if solved else 'not solved'})")

    controls = extract_path(best_v)
    if len(controls) == 0:
        print("\nno edge was ever added -- nothing to dump")
        return
    rows = replay_and_dump(sys_, controls, args.out)
    carts = [r["cart"] for r in rows]
    print(f"\nsolution path: {len(controls)} control steps, "
          f"{len(controls)*sys_.dt:.2f}s")
    print(f"  cart: start {carts[0]:+.3f}  max {max(carts):+.3f}  "
          f"final {carts[-1]:+.3f}")
    print(f"  min clearance: {min(r['min_clearance'] for r in rows):+.4f} m")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
