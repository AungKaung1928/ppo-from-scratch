"""
Step 6 -- the same cart-pole, integrated by MuJoCo, with a continuous actuator.

WHY THIS IS A DIFFERENT ENVIRONMENT AND IS TREATED AS ONE
---------------------------------------------------------
Steps 1-5 measure a numpy plant with explicit Euler and two actions. This file
hands the same physical parameters to MuJoCo and opens the actuator up to the
whole interval. Two things change at once, so both are pinned down separately:

  the plant     -- identical. cartpole_mj.xml sets the inertias by hand rather
                   than letting MuJoCo derive them from a capsule geom, so the
                   uniform-thin-rod assumption behind the 4/3 in cartpole.py
                   holds exactly. Check V1 compares MuJoCo's qacc against
                   cartpole.accelerations() over random states: they agree to
                   ~1e-14 absolute. The two simulators are describing the same
                   object.

  the integrator -- different, and NOT reconcilable. MuJoCo's "Euler" is
                   semi-implicit: it updates velocity first and then advances
                   position with the NEW velocity. cartpole.py deliberately
                   uses the explicit ordering, because that is what CartPole-v1
                   does. Check V2 measures the resulting one-step gap instead of
                   pretending it away.

  the actions   -- continuous, ctrl in [-1, 1] with gear 10, so the endpoints
                   are exactly the +-10 N the discrete env applies. Same
                   actuator authority. The only new freedom is everything in
                   between, which is precisely the thing a Gaussian policy can
                   use and a categorical one cannot.

Observation ordering is kept as [x, x_dot, theta, theta_dot] so the step-2 LQR
gain applies to this plant without reindexing -- which is what check V5 uses to
ask whether a controller designed against the numpy plant survives being moved
to a different integrator. That is sim-to-real in miniature, with the reality
gap reduced to one term.
"""

import math
import time

import numpy as np

import cartpole as cp
from cartpole import (INIT_RANGE, MAX_EPISODE_STEPS, THETA, THETA_DOT,
                      THETA_THRESHOLD, X, X_DOT, X_THRESHOLD)

MODEL_PATH = "cartpole_mj.xml"
GEAR = 10.0

_RESULTS = []


def record(name, passed, detail):
    _RESULTS.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:<50} {detail}")


def note(name, detail):
    print(f"  [ -- ] {name:<50} {detail}")


class MjCartPoleEnv:
    """Continuous-action cart-pole on MuJoCo. Same reward, same termination
    bounds, same 500-step cap and same +-0.05 init box as CartPoleEnv, so
    'solved' means the same thing it means in steps 1-5."""

    obs_dim = 4
    act_dim = 1
    action_low, action_high = -1.0, 1.0

    def __init__(self, max_episode_steps=MAX_EPISODE_STEPS, seed=None,
                 model_path=MODEL_PATH):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._done = True

    def _obs(self):
        return np.array([self.data.qpos[0], self.data.qvel[0],
                         self.data.qpos[1], self.data.qvel[1]], dtype=np.float64)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        s = self._rng.uniform(-INIT_RANGE, INIT_RANGE, size=4)
        self.data.qpos[:] = [s[X], s[THETA]]
        self.data.qvel[:] = [s[X_DOT], s[THETA_DOT]]
        self._mj.mj_forward(self.model, self.data)
        self._steps, self._done = 0, False
        return self._obs(), {}

    def step(self, action):
        if self._done:
            raise RuntimeError("step() after the episode ended. Call reset() first.")
        a = float(np.clip(np.asarray(action).reshape(-1)[0],
                          self.action_low, self.action_high))
        self.data.ctrl[0] = a
        self._mj.mj_step(self.model, self.data)
        self._steps += 1
        obs = self._obs()
        terminated = bool(abs(obs[X]) > X_THRESHOLD or abs(obs[THETA]) > THETA_THRESHOLD)
        truncated = bool(not terminated and self._steps >= self.max_episode_steps)
        self._done = terminated or truncated
        return obs, 1.0, terminated, truncated, {}


# ------------------------------------------------------------------ checks --

def check_V1():
    import mujoco
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    worst_abs = worst_rel = 0.0
    for _ in range(5000):
        s = np.array([rng.uniform(-2, 2), rng.uniform(-3, 3),
                      rng.uniform(-0.6, 0.6), rng.uniform(-4, 4)])
        F = rng.uniform(-10, 10)
        d.qpos[:] = [s[X], s[THETA]]
        d.qvel[:] = [s[X_DOT], s[THETA_DOT]]
        d.ctrl[0] = F / GEAR
        mujoco.mj_forward(m, d)
        xa, ta = cp.accelerations(s, F)
        e = np.abs(d.qacc - np.array([xa, ta]))
        worst_abs = max(worst_abs, float(e.max()))
        worst_rel = max(worst_rel, float((e / np.maximum(np.abs([xa, ta]), 1e-9)).max()))
    record("V1. MuJoCo qacc == cartpole.accelerations()", worst_abs < 1e-11,
           f"max abs {worst_abs:.3e}, max rel {worst_rel:.3e}, 5000 states")
    note("      pole inertia set by hand, not from the geom",
         f"I_transverse {m.body_inertia[2][0]:.10f} vs m L^2/12 = {0.1/12:.10f}")


def check_V2():
    """MuJoCo's Euler is semi-implicit; ours is explicit. Measure the gap."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(1)
    dth, dx = [], []
    for _ in range(5000):
        s = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1),
                      rng.uniform(-THETA_THRESHOLD, THETA_THRESHOLD),
                      rng.uniform(-1, 1)])
        F = rng.choice([-10.0, 10.0])
        d.qpos[:] = [s[X], s[THETA]]
        d.qvel[:] = [s[X_DOT], s[THETA_DOT]]
        d.ctrl[0] = F / GEAR
        mujoco.mj_step(m, d)
        ours = cp.euler_step(s, F)
        dth.append(abs(d.qpos[1] - ours[THETA]))
        dx.append(abs(d.qpos[0] - ours[X]))
    dth, dx = np.array(dth), np.array(dx)
    note("V2. one-step gap, MuJoCo semi-implicit vs our explicit Euler",
         f"median |dtheta| {np.median(dth):.3e} rad, max {dth.max():.3e} "
         f"({dth.max()/THETA_THRESHOLD:.2%} of the limit)")
    note("      and in cart position",
         f"median |dx| {np.median(dx):.3e} m, max {dx.max():.3e} m")
    note("      why it is not reconcilable",
         "semi-implicit advances x with the NEW x_dot; v1 uses the old one")


def check_V3():
    """Energy under zero control. Semi-implicit Euler is symplectic-ish and
    should drift far less than the explicit version measured in step 1."""
    env = MjCartPoleEnv(max_episode_steps=10_000, seed=0)
    env.reset()
    env.data.qpos[:] = [0.0, 0.5]
    env.data.qvel[:] = [0.0, 0.0]
    env._mj.mj_forward(env.model, env.data)
    e0 = cp.energy(env._obs())
    worst50 = worst500 = 0.0
    for k in range(1, 501):
        env.data.ctrl[0] = 0.0
        env._mj.mj_step(env.model, env.data)
        rel = abs(cp.energy(env._obs()) - e0) / abs(e0)
        if k <= 50:
            worst50 = max(worst50, rel)
        worst500 = max(worst500, rel)
    note("V3. energy drift, zero control, theta0 = 0.5 rad",
         f"{worst50:+.3%} at 50 steps, {worst500:+.3%} at 500")
    note("      step 1 measured, same test, explicit Euler",
         "+0.427% at 50 steps, +145.0% at 500")


def check_V4():
    env = MjCartPoleEnv(max_episode_steps=5, seed=3)
    env.reset()
    rs, terms, truncs = [], [], []
    for _ in range(5):
        _, r, te, tr, _ = env.step(0.0)
        rs.append(r); terms.append(te); truncs.append(tr)
    ok = (rs == [1.0] * 5 and terms == [False] * 5
          and truncs == [False, False, False, False, True])
    record("V4a. truncation at exactly the cap, reward 1.0 each step", ok,
           f"rewards {rs}, truncated {truncs}")
    try:
        env.step(0.0)
        raised = False
    except RuntimeError:
        raised = True
    record("V4b. step() after the episode ended raises", raised, "RuntimeError")

    a = MjCartPoleEnv(seed=11); b = MjCartPoleEnv(seed=11); c = MjCartPoleEnv(seed=12)
    oa, _ = a.reset(); ob, _ = b.reset(); oc, _ = c.reset()
    for _ in range(10):   # 10 steps only: at lambda ~ 4/s the pole leaves the
        # 12 deg window in well under 50 steps under a constant action, and a
        # determinism check must not be an episode-termination check.
        oa, *_ = a.step(0.05); ob, *_ = b.step(0.05); oc, *_ = c.step(0.05)
    record("V4c. same seed bitwise identical, different seed differs",
           np.array_equal(oa, ob) and not np.array_equal(oa, oc),
           f"max |a-b| {np.max(np.abs(oa-ob)):.1e}, |a-c| {np.max(np.abs(oa-oc)):.1e}")

    env = MjCartPoleEnv(seed=5)
    rng = np.random.default_rng(5)
    rets = []
    for _ in range(2000):
        env.reset(); tot = 0.0
        while True:
            _, r, te, tr, _ = env.step(rng.uniform(-1, 1))
            tot += r
            if te or tr:
                break
        rets.append(tot)
    rets = np.array(rets)
    note("V4d. uniform-random continuous policy, 2000 episodes",
         f"mean {rets.mean():.2f} +- {rets.std()/math.sqrt(len(rets)):.2f}, "
         f"median {np.median(rets):.0f}")
    note("      the discrete env, step 1 check 6", "22.09 +- 0.12 over 10k episodes")


def check_V5():
    """Does the step-2 LQR design survive being moved to a different
    integrator? The gain was computed against the numpy plant and is applied
    here unchanged."""
    import lqr as L
    Ac, Bc = L.linearise_analytic()
    Ad, Bd = L.discretise(Ac, Bc, cp.DEFAULT.tau)
    Q, R = np.eye(4), np.array([[1.0]])
    P, _, _ = L.dare_iterate(Ad, Bd, Q, R)
    K = L.lqr_gain(Ad, Bd, Q, R, P)

    for name, project in (("bang-bang (sign only, +-10 N)",
                           lambda u: 1.0 if u > 0 else -1.0),
                          ("continuous (clipped to +-10 N)",
                           lambda u: float(np.clip(u / GEAR, -1.0, 1.0)))):
        env = MjCartPoleEnv(seed=0)
        rets = []
        for _ in range(100):
            obs, _ = env.reset(); tot = 0.0
            while True:
                obs, r, te, tr, _ = env.step(project(-(K @ obs).item()))
                tot += r
                if te or tr:
                    break
            rets.append(tot)
        rets = np.array(rets)
        record(f"V5. step-2 LQR gain on the MuJoCo plant, {name}",
               rets.mean() >= 475.0,
               f"mean {rets.mean():.1f} over 100 episodes (min {rets.min():.0f})")


def check_V6():
    env = MjCartPoleEnv(seed=0)
    env.reset()
    n, t0 = 20_000, time.perf_counter()
    for _ in range(n):
        _, _, te, tr, _ = env.step(0.0)
        if te or tr:
            env.reset()
    rate = n / (time.perf_counter() - t0)
    note("V6. throughput", f"{rate:,.0f} env steps/s "
         f"(numpy env, step 1: 373,919/s -- MuJoCo is {373919/rate:.0f}x slower)")
    note("      cost of a 150k-step run in env time alone", f"{150_000/rate:.1f} s")


def main():
    print("\n=== step 6: MuJoCo continuous-action cart-pole ===\n")
    check_V1(); check_V2(); check_V3(); check_V4(); check_V5(); check_V6()
    ok = all(_RESULTS)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed"
          f"{'' if ok else '  <-- FAILURES'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
