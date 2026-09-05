"""Verification harness for `cartpole.py`. Run before trusting any RL result.

    python verify_env.py

Seven checks. Read what each one does and does NOT prove -- that distinction is
the whole value of this file:

  1. Independent derivation. The mass matrix is assembled from the Lagrangian and
     inverted numerically; the env's hand-eliminated scalar formulas must agree.
     Shares no algebra with the env. Catches sign errors and bad elimination.
     Does NOT catch an error made identically in both derivations.
  2. Conserved quantities. Energy and horizontal momentum, both written straight
     from the Lagrangian, must be invariant under RK4 with F = 0. Catches a
     dropped or mis-scaled term in the accelerations. Proves the accelerations
     ARE the Euler-Lagrange equations of the stated energy -- internal
     consistency, not external correctness.
  3. Closed-form limit. With the cart pinned, the linearised inverted pendulum
     must diverge as cosh(lambda t) with lambda = sqrt(3g/4l) = 3.8341 s^-1.
     This number comes from outside the code, so it is the check that pins the
     half-length convention and the 4/3.
  4. Episode contract. Thresholds, reward on the terminal step, truncation at
     exactly the cap, terminated/truncated kept apart, no stepping past the end.
  5. Determinism and RNG isolation. Same seed, same trajectory; the global numpy
     RNG must not be able to move the env.
  6. Random-policy return. Must land near the widely reported ~22 steps. Only
     external cross-check available without gymnasium installed.
  7. Golden trajectory. Regression, NOT correctness -- it freezes today's
     behaviour so a refactor during steps 3-5 cannot silently move the env.

Known gap, stated plainly: gymnasium is not installed, so there is no bit-diff
against the reference implementation. Checks 1-3 are independent derivations and
check 6 is the only externally published number. Nothing here proves equality
with CartPole-v1; it proves the equations are the published ones and are
self-consistent.
"""
import math
import os
import sys

import numpy as np

from cartpole import (
    DEFAULT,
    MAX_EPISODE_STEPS,
    THETA,
    THETA_DOT,
    THETA_THRESHOLD,
    X,
    X_THRESHOLD,
    CartPoleEnv,
    CartPoleParams,
    accelerations,
    deriv,
    energy,
    euler_step,
    momentum,
)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_traj.npz")

_results = []


def record(name, passed, detail):
    _results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n         {detail}")


def note(name, detail):
    print(f"  [ -- ] {name}\n         {detail}")


# --------------------------------------------------------------------------
# check 1 -- independent derivation via the Lagrangian mass matrix
# --------------------------------------------------------------------------
def accel_matrix(state, force, p=DEFAULT):
    """Solve M(q) qddot = f(q, qdot, u) instead of using the eliminated formulas.

        [ M+m        m l cos(th) ] [ xddot  ]   [ F + m l thdot^2 sin(th) ]
        [ m l cos(th)  4/3 m l^2 ] [ thddot ] = [ m g l sin(th)           ]

    Row 1 is the horizontal Newton equation, row 2 the moment equation about the
    pole's pivot. Assembled from the Lagrangian directly; no algebra is shared
    with `cartpole.accelerations`.
    """
    theta, theta_dot = state[THETA], state[THETA_DOT]
    c, s = math.cos(theta), math.sin(theta)
    m, l = p.masspole, p.length

    M = np.array([[p.total_mass, m * l * c], [m * l * c, (4.0 / 3.0) * m * l ** 2]])
    f = np.array([force + m * l * theta_dot ** 2 * s, m * p.gravity * l * s])
    xacc, thetaacc = np.linalg.solve(M, f)
    return xacc, thetaacc


def check_1_derivation():
    rng = np.random.default_rng(0)
    n = 10_000
    worst_abs = worst_rel = 0.0
    # deliberately wider than the env ever reaches, so the agreement is not an
    # artefact of small angles
    for _ in range(n):
        state = np.array(
            [
                rng.uniform(-2.4, 2.4),
                rng.uniform(-5, 5),
                rng.uniform(-math.pi, math.pi),
                rng.uniform(-10, 10),
            ]
        )
        force = rng.uniform(-10, 10)
        a = np.array(accelerations(state, force))
        b = np.array(accel_matrix(state, force))
        d = np.abs(a - b)
        worst_abs = max(worst_abs, d.max())
        worst_rel = max(worst_rel, (d / (1.0 + np.abs(b))).max())
    record(
        "1. scalar formulas == Lagrangian mass-matrix solve",
        worst_rel < 1e-12,
        f"{n} random states, |theta| up to pi: max abs diff {worst_abs:.3e}, "
        f"max rel diff {worst_rel:.3e} (tol 1e-12)",
    )


# --------------------------------------------------------------------------
# check 2 -- conserved quantities under RK4 with no input
# --------------------------------------------------------------------------
def rk4(state, force, dt, p=DEFAULT):
    k1 = deriv(state, force, p)
    k2 = deriv(state + 0.5 * dt * k1, force, p)
    k3 = deriv(state + 0.5 * dt * k2, force, p)
    k4 = deriv(state + dt * k3, force, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def check_2_conservation():
    dt, T = 1e-4, 2.0
    s = np.array([0.0, 0.0, 0.5, 0.0])          # released from 0.5 rad, cart free
    e0, p0 = energy(s), momentum(s)
    worst_e = worst_p = 0.0
    for _ in range(int(T / dt)):
        s = rk4(s, 0.0, dt)
        worst_e = max(worst_e, abs(energy(s) - e0) / abs(e0))
        worst_p = max(worst_p, abs(momentum(s) - p0))
    record(
        "2a. energy conserved, F=0, RK4",
        worst_e < 1e-6,
        f"released from theta=0.5 rad, {T} s at dt={dt}: max relative drift "
        f"{worst_e:.3e} (tol 1e-6)",
    )
    record(
        "2b. horizontal momentum conserved, F=0, RK4",
        worst_p < 1e-9,
        f"p0 = {p0:.3e} kg m/s, max absolute drift {worst_p:.3e} (tol 1e-9)",
    )


# --------------------------------------------------------------------------
# check 3 -- fixed-cart limit against a closed form from outside the code
# --------------------------------------------------------------------------
def check_3_pinned_cart():
    p = CartPoleParams(masscart=1e9)            # cart effectively immovable
    lam_true = math.sqrt(3 * p.gravity / (4 * p.length))
    th0, dt, T = 1e-6, 1e-4, 1.0
    s = np.array([0.0, 0.0, th0, 0.0])
    for _ in range(int(T / dt)):
        s = rk4(s, 0.0, dt, p)
    lam_meas = math.acosh(s[THETA] / th0) / T
    rel = abs(lam_meas - lam_true) / lam_true
    record(
        "3. pinned-cart divergence rate == sqrt(3g/4l)",
        rel < 5e-3,
        f"lambda measured {lam_meas:.6f} s^-1 vs closed form {lam_true:.6f} s^-1, "
        f"rel err {rel:.2e} (tol 5e-3). Pins the HALF-length convention and the 4/3.",
    )


# --------------------------------------------------------------------------
# check 4 -- episode contract
# --------------------------------------------------------------------------
def check_4_contract():
    # 4a: truncation fires at exactly the cap, and is not termination
    env = CartPoleEnv(max_episode_steps=5, seed=0)
    env.reset()
    flags = [env.step(0)[2:4] for _ in range(5)]
    ok = flags[:4] == [(False, False)] * 4 and flags[4] == (False, True)
    record(
        "4a. truncation at exactly the cap, terminated stays False",
        ok,
        f"cap=5, per-step (terminated, truncated) = {flags}",
    )

    # 4b: stepping a finished episode raises
    try:
        env.step(0)
        raised = False
    except RuntimeError:
        raised = True
    record("4b. step() after episode end raises", raised, "RuntimeError as designed")

    # 4c: reward is 1.0 on every step including the terminal one
    env = CartPoleEnv(seed=1)
    env.reset()
    rewards, term = [], False
    while not term:
        _, r, term, trunc, _ = env.step(1)          # always push right -> fails fast
        rewards.append(r)
        if trunc:
            break
    record(
        "4c. reward == 1.0 on every step, terminal included",
        all(r == 1.0 for r in rewards) and term,
        f"always-right policy terminated after {len(rewards)} steps, "
        f"return {sum(rewards)}",
    )

    # 4d: termination bounds are the v1 bounds
    env = CartPoleEnv(seed=2)
    env.reset()
    env.state = np.array([0.0, 0.0, THETA_THRESHOLD - 1e-9, 0.0])
    _, _, t_in, _, _ = env.step(1)                  # tiny step, still inside
    env2 = CartPoleEnv(seed=2)
    env2.reset()
    env2.state = np.array([X_THRESHOLD + 1e-6, 0.0, 0.0, 0.0])
    _, _, t_out, _, _ = env2.step(1)
    record(
        "4d. termination bounds are |x|>2.4 and |theta|>12 deg",
        (not t_in) and t_out,
        f"theta_threshold {THETA_THRESHOLD:.17f} rad, x_threshold {X_THRESHOLD}",
    )

    # 4e: sign convention -- a trivial bang-bang law must beat random by an order
    # of magnitude. If the action->force sign were flipped this collapses to ~22.
    lens = []
    env = CartPoleEnv(seed=3)
    for _ in range(100):
        s, _ = env.reset()
        done, n = False, 0
        while not done:
            a = 1 if (s[THETA] + 0.5 * s[THETA_DOT]) > 0 else 0
            s, _, term, trunc, _ = env.step(a)
            done = term or trunc
            n += 1
        lens.append(n)
    mean_len = float(np.mean(lens))
    record(
        "4e. action sign: bang-bang theta+0.5*thetadot balances",
        mean_len > 100,
        f"mean return {mean_len:.1f} over 100 episodes "
        f"(min {min(lens)}, max {max(lens)}). A flipped force sign scores ~22.",
    )


# --------------------------------------------------------------------------
# check 5 -- determinism and RNG isolation
# --------------------------------------------------------------------------
def check_5_determinism():
    def rollout(seed, actions):
        env = CartPoleEnv(seed=seed)
        s, _ = env.reset()
        traj = [s]
        for a in actions:
            s, _, term, trunc, _ = env.step(a)
            traj.append(s)
            if term or trunc:
                break
        return np.array(traj)

    acts = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0] * 2
    a1, a2 = rollout(7, acts), rollout(7, acts)
    same = a1.shape == a2.shape and np.array_equal(a1, a2)
    b = rollout(8, acts)
    differs = not np.array_equal(a1[0], b[0])
    record(
        "5a. same seed -> bitwise identical trajectory",
        same and differs,
        f"seed 7 twice: identical={same}; seed 8 initial state differs={differs}",
    )

    np.random.seed(1234)
    c1 = rollout(7, acts)
    np.random.seed(999)
    c2 = rollout(7, acts)
    record(
        "5b. env ignores the global numpy RNG",
        np.array_equal(c1, c2),
        "global np.random.seed changed between rollouts, trajectory unchanged",
    )


# --------------------------------------------------------------------------
# check 6 -- random policy against the published ~22
# --------------------------------------------------------------------------
def check_6_random_policy():
    env = CartPoleEnv(seed=0)
    rng = np.random.default_rng(0)
    n = 10_000
    lens = np.empty(n)
    for i in range(n):
        env.reset()
        done, k = False, 0
        while not done:
            _, _, term, trunc, _ = env.step(int(rng.integers(2)))
            done = term or trunc
            k += 1
        lens[i] = k
    mean = lens.mean()
    se = lens.std(ddof=1) / math.sqrt(n)
    record(
        "6. random-policy mean return near the published ~22",
        18.0 < mean < 26.0,
        f"{n} episodes: mean {mean:.2f} +/- {se:.2f} (SE), median {np.median(lens):.0f}, "
        f"min {lens.min():.0f}, max {lens.max():.0f}",
    )


# --------------------------------------------------------------------------
# check 7 -- golden trajectory regression
# --------------------------------------------------------------------------
def check_7_golden():
    env = CartPoleEnv(seed=42)
    s, _ = env.reset()
    rng = np.random.default_rng(42)
    traj = [s]
    for _ in range(200):
        s, _, term, trunc, _ = env.step(int(rng.integers(2)))
        traj.append(s)
        if term or trunc:
            env.reset()
            s = env.state.copy()
    traj = np.array(traj)

    if not os.path.exists(GOLDEN):
        np.savez_compressed(GOLDEN, traj=traj)
        note("7. golden trajectory", f"created {os.path.basename(GOLDEN)}, {traj.shape} -- commit it")
        return
    ref = np.load(GOLDEN)["traj"]
    if ref.shape != traj.shape:
        record("7. golden trajectory regression", False, f"shape {traj.shape} vs stored {ref.shape}")
        return
    d = np.abs(ref - traj).max()
    record(
        "7. golden trajectory regression",
        d < 1e-12,
        f"{traj.shape[0]} states, max abs diff {d:.3e}, "
        f"bitwise identical={np.array_equal(ref, traj)} (regression only, not correctness)",
    )


# --------------------------------------------------------------------------
# reported numbers -- measurements, not pass/fail
# --------------------------------------------------------------------------
def report_numbers():
    """Measurements the README quotes. No pass/fail -- these characterise the env."""
    print("\nReported numbers (not pass/fail):")

    # 1. Integrator error. Reported as one-step local error plus a SHORT-horizon
    #    open-loop divergence. A full-episode open-loop comparison is not reported
    #    because it is meaningless: check 3 measured lambda = 3.834 s^-1, so over a
    #    10 s episode any difference between two integrators is amplified by
    #    e^38 ~ 3e16 and the number says nothing about integrator quality.
    p = DEFAULT
    rng = np.random.default_rng(0)
    n = 10_000
    err = np.empty((n, 2))
    for i in range(n):
        s = np.array([rng.uniform(-2.4, 2.4), rng.uniform(-3, 3),
                      rng.uniform(-THETA_THRESHOLD, THETA_THRESHOLD), rng.uniform(-3, 3)])
        f = p.force_mag if rng.integers(2) else -p.force_mag
        ref = s.copy()
        for _ in range(20):
            ref = rk4(ref, f, p.tau / 20, p)
        d = np.abs(euler_step(s, f, p) - ref)
        err[i] = (d[THETA], d[X])
    note(
        "one-step Euler error vs RK4, inside the operational envelope",
        f"{n} random states: median |d theta| {np.median(err[:,0]):.2e} rad, "
        f"max {err[:,0].max():.2e} rad ({100*err[:,0].max()/THETA_THRESHOLD:.2f}% of the "
        f"12 deg limit); median |d x| {np.median(err[:,1]):.2e} m, max {err[:,1].max():.2e} m.",
    )

    s_e = np.array([0.0, 0.0, 0.02, 0.0])
    s_r, forces = s_e.copy(), []
    for _ in range(25):                   # 0.5 s, e^(lambda t) ~ 6.8, still readable
        f = p.force_mag if (s_e[THETA] + 0.5 * s_e[THETA_DOT]) > 0 else -p.force_mag
        forces.append(f)
        s_e = euler_step(s_e, f, p)
        for _ in range(20):
            s_r = rk4(s_r, f, p.tau / 20, p)
    note(
        "Euler vs RK4, 25 controlled steps (0.5 s), identical force sequence",
        f"|d theta| {abs(s_e[THETA]-s_r[THETA]):.2e} rad "
        f"({100*abs(s_e[THETA]-s_r[THETA])/THETA_THRESHOLD:.2f}% of the limit), "
        f"|d x| {abs(s_e[X]-s_r[X]):.2e} m. The v1 integrator is a slightly different "
        f"plant from the true ODE and is kept anyway, so that 475/500 means what it "
        f"means in every other published CartPole-v1 result.",
    )

    # 2. Energy invented in free fall. Reported at two horizons because the long
    #    one leaves the operational envelope and overstates the error.
    s0 = np.array([0.0, 0.0, 0.05, 0.0])
    e0 = energy(s0)
    s, drift = s0.copy(), {}
    for k in range(1, MAX_EPISODE_STEPS + 1):
        s = euler_step(s, 0.0, p)
        if k in (50, MAX_EPISODE_STEPS):
            drift[k] = 100 * (energy(s) - e0) / abs(e0)
    note(
        "explicit-Euler energy gain, F=0, from theta=0.05 rad",
        f"E0 {e0:.6f} J; +{drift[50]:.3f}% after 50 steps (still inside 12 deg), "
        f"+{drift[MAX_EPISODE_STEPS]:.1f}% after 500 (pole has fallen right over, so "
        f"the long-horizon figure overstates what an episode sees).",
    )

    # 3. Throughput. This measurement overturned a planning assumption -- see README.
    import time
    env = CartPoleEnv(seed=0)
    env.reset()
    n, t0 = 200_000, time.perf_counter()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(1)
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    rate = n / dt
    note(
        "scalar env throughput, single core, nice -n 10",
        f"{rate:,.0f} steps/s. A 200k-step PPO run therefore spends {200_000/rate:.1f} s "
        f"total inside the env. That is ~1% of the 10-minute budget, so the env is NOT "
        f"the bottleneck and the vectorised rewrite planned for step 3 is not needed to "
        f"meet the budget. torch forward/backward will dominate. Measure before writing it.",
    )


def main():
    print("cartpole.py verification\n" + "=" * 60)
    check_1_derivation()
    check_2_conservation()
    check_3_pinned_cart()
    check_4_contract()
    check_5_determinism()
    check_6_random_policy()
    check_7_golden()
    report_numbers()

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 60)
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        return 1
    print(f"all {len(_results)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
