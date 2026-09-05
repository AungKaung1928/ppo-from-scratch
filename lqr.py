"""
Step 2 -- LQR baseline for cart-pole.

WHY THIS EXISTS
---------------
Before claiming a learned policy is worth anything, there has to be a number it
must beat. For a plant whose equations we already wrote down by hand in
cartpole.py, the honest baseline is not "random" and not "a tuned heuristic" --
it is the optimal linear controller for the linearised plant, designed from the
model alone, at a sample cost of exactly zero environment steps.

If PPO needs 10^5 environment steps to match something that needed 0, that is
the finding. It does not make PPO useless -- it makes the *reason* you would
reach for PPO explicit: you reach for it when you do not have the model.

THE THREE PIECES
----------------
1. Linearisation about the upright equilibrium (0, 0, 0, 0).

   From cartpole.accelerations(), with s -> theta, c -> 1, and dropping the
   theta_dot**2 * sin(theta) term (second order in the state):

       D      = l * (4/3 - m/M)                     [m]
       thddot = (g/D) * theta - (1/(M*D)) * F
       xddot  = F/M - (m*l/M) * thddot

   so the continuous system is xdot = Ac x + Bc F with only four nonzero
   entries in Ac and two in Bc. Derived by hand above, and checked in code
   against a central-difference Jacobian of cartpole.deriv(). Two paths, same
   discipline as step 1.

2. Discretisation. This is the one place the "ugly" explicit-Euler integrator
   we kept in step 1 pays for itself. Every acceleration term in euler_step()
   is evaluated at the OLD state, so the full update is exactly

       s' = s + tau * f(s, u)

   and therefore the linearised discrete map is exactly

       Ad = I + tau*Ac,   Bd = tau*Bc

   with no approximation whatsoever. A semi-implicit or exact-ZOH plant would
   need expm([[Ac,Bc],[0,0]]*tau) here. We do not, because our plant really is
   forward Euler. Checked numerically against euler_step() all the same.

3. Discrete-time infinite-horizon LQR, solved by iterating the Riccati
   recursion to a fixed point:

       P_{k+1} = Q + Ad' P_k Ad
                   - Ad' P_k Bd (R + Bd' P_k Bd)^-1 Bd' P_k Ad,   P_0 = Q
       K       = (R + Bd' P Bd)^-1 Bd' P Ad,   u = -K x

   No scipy, no solve_discrete_are. The recursion IS the dynamic-programming
   backup for a quadratic value function: P_k is the cost-to-go matrix with k
   steps left, and iterating it to convergence is value iteration in closed
   form. That is the whole reason to write it out rather than call a library --
   it is the same object PPO's critic is trying to learn by regression, except
   here we can compute it exactly.

THE PART THAT IS NOT TEXTBOOK
-----------------------------
Our env has TWO actions, +-10 N. LQR hands back a continuous u. Something has
to bridge that, and the choice is a design decision, not a detail:

    action = 1 if u > 0 else 0          i.e. F = force_mag * sign(-Kx)

This is bang-bang control with an LQR-derived switching surface. Only the SIGN
of -Kx reaches the plant, so the policy is invariant to any positive rescaling
of K -- and therefore to joint scaling of the cost, (Q,R) -> (cQ,cR), which
leaves K bitwise identical (L6).

I first wrote that R ALONE only rescales K and so could not matter. That is
wrong and the check caught it: the DARE is not linear in R, and raising R from
0.01 to 100 rotates the normalised gain direction by 0.13. What is true is the
weaker and more interesting statement -- the rotation changes the measured
return by exactly nothing, because every one of these controllers scores
500.0/500. Return on cart-pole is a saturating metric. It cannot rank
controllers that are all merely good enough, which is precisely why steps 4
and 5 rank on steps-to-threshold and not on return.

We report both the sign-projected controller (the real baseline, same action
space as PPO) and a continuous-force controller (what LQR would do if it were
allowed real actuators), so that "LQR is good" and "bang-bang is good" stay
separable.

VERIFICATION IN THIS FILE
-------------------------
    L1  analytic Ac,Bc == central-difference Jacobian of deriv()
    L2  Ad,Bd == I+tau*Ac, tau*Bc == finite-difference Jacobian of euler_step()
    L3  converged P satisfies the DARE residual ~ 0
    L4  closed-loop spectral radius of (Ad - Bd K) < 1
    L5  x'Px == simulated linear closed-loop cost-to-go (independent check on P)
    L6  cost-scaling invariance: (cQ, cR) gives an identical gain

What none of it proves: that the linear model is a good description of the
plant far from upright. It is not, and the basin sweep measures exactly where
it stops being one -- by bisecting on initial VELOCITY, because sweeping
initial angle finds nothing: every angle inside the termination limit is
recoverable, so that sweep has no information in it.
"""

import json
import math
import time

import numpy as np

import cartpole as cp
from cartpole import (DEFAULT, MAX_EPISODE_STEPS, THETA, THETA_DOT, X, X_DOT,
                      CartPoleEnv, CartPoleParams)

# ---------------------------------------------------------------- reporting --

_RESULTS = []


def record(name, passed, detail):
    _RESULTS.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:<52} {detail}")


def note(name, detail):
    print(f"  [ -- ] {name:<52} {detail}")


# ------------------------------------------------------------ linearisation --

def linearise_analytic(p=DEFAULT):
    """Continuous Ac, Bc about the upright equilibrium. Derived by hand."""
    M, m, l, g = p.total_mass, p.masspole, p.length, p.gravity
    D = l * (4.0 / 3.0 - m / M)

    a43 = g / D                       # d(thetaacc)/d(theta)
    b4 = -1.0 / (M * D)               # d(thetaacc)/d(F)
    a23 = -(m * l / M) * a43          # d(xacc)/d(theta)
    b2 = 1.0 / M - (m * l / M) * b4   # d(xacc)/d(F)

    Ac = np.zeros((4, 4))
    Ac[X, X_DOT] = 1.0
    Ac[THETA, THETA_DOT] = 1.0
    Ac[X_DOT, THETA] = a23
    Ac[THETA_DOT, THETA] = a43

    Bc = np.zeros((4, 1))
    Bc[X_DOT, 0] = b2
    Bc[THETA_DOT, 0] = b4
    return Ac, Bc


def jacobian(fn, s0, u0, eps=1e-6):
    """Central-difference Jacobians of fn(state, force) wrt state and force."""
    A = np.zeros((4, 4))
    for j in range(4):
        sp, sm = s0.copy(), s0.copy()
        sp[j] += eps
        sm[j] -= eps
        A[:, j] = (fn(sp, u0) - fn(sm, u0)) / (2 * eps)
    B = ((fn(s0, u0 + eps) - fn(s0, u0 - eps)) / (2 * eps)).reshape(4, 1)
    return A, B


def discretise(Ac, Bc, tau):
    """Exact for a forward-Euler plant. See module docstring, piece 2."""
    return np.eye(4) + tau * Ac, tau * Bc


# ---------------------------------------------------------------- riccati ----

def dare_iterate(A, B, Q, R, tol=1e-13, max_iter=200_000):
    """Iterate the discrete Riccati recursion to a fixed point.

    P_0 = Q is the terminal cost; each sweep adds one step of horizon, so this
    is finite-horizon dynamic programming run until the horizon stops mattering.
    Converges linearly at rate rho(Ad - Bd K)^2 when (A,B) is stabilisable and
    (A, Q^1/2) detectable -- both hold here.

    The tolerance is RELATIVE to ||P||_inf. It was absolute at 1e-14 at first,
    and that quietly turned the whole thing into a coin flip: P has entries near
    7e3, so an absolute 1e-14 sits below the float64 resolution of P itself
    (eps * 7e3 ~ 1.5e-12). Q = I4 happened to land on an exact fixed point and
    converged; Q = diag(10,10,1,1) chattered in the last bit forever and looked
    like a stabilisability failure. It was not. It was the tolerance.
    """
    P = Q.copy().astype(np.float64)
    for k in range(1, max_iter + 1):
        BtP = B.T @ P
        S = R + BtP @ B
        K = np.linalg.solve(S, BtP @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        P_next = 0.5 * (P_next + P_next.T)      # kill asymmetry drift
        delta = np.max(np.abs(P_next - P))
        scale = max(1.0, float(np.max(np.abs(P_next))))
        P = P_next
        if delta < tol * scale:
            return P, k, delta / scale
    raise RuntimeError(f"Riccati recursion did not converge in {max_iter} sweeps")


def dare_residual(A, B, Q, R, P):
    BtP = B.T @ P
    S = R + BtP @ B
    K = np.linalg.solve(S, BtP @ A)
    return float(np.max(np.abs(Q + A.T @ P @ A - A.T @ P @ B @ K - P)))


def lqr_gain(A, B, Q, R, P):
    BtP = B.T @ P
    return np.linalg.solve(R + BtP @ B, BtP @ A)      # (1,4)


# ---------------------------------------------------------------- policies --

def make_bangbang_policy(K):
    """LQR switching surface projected onto the two available actions."""
    def policy(obs):
        u = -(K @ obs).item()
        return 1 if u > 0.0 else 0
    return policy


def evaluate(policy, episodes=100, seed=0, params=DEFAULT,
             max_episode_steps=MAX_EPISODE_STEPS):
    env = CartPoleEnv(params=params, max_episode_steps=max_episode_steps, seed=seed)
    returns = np.empty(episodes)
    for i in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        while True:
            obs, r, terminated, truncated, _ = env.step(policy(obs))
            total += r
            if terminated or truncated:
                break
        returns[i] = total
    return returns


def rollout_continuous(K, s0, params=DEFAULT, max_steps=MAX_EPISODE_STEPS, clip=None):
    """Same plant, same termination, but real-valued force. Not the baseline --
    only here so 'LQR is good' and 'bang-bang is good' stay separable."""
    s = np.asarray(s0, dtype=np.float64).copy()
    for t in range(max_steps):
        u = -(K @ s).item()
        if clip is not None:
            u = max(-clip, min(clip, u))
        s = cp.euler_step(s, u, params)
        if abs(s[X]) > cp.X_THRESHOLD or abs(s[THETA]) > cp.THETA_THRESHOLD:
            return t + 1
    return max_steps


# ------------------------------------------------------------------ checks --

def check_L1_L2(p=DEFAULT):
    Ac, Bc = linearise_analytic(p)
    Ac_n, Bc_n = jacobian(lambda s, u: cp.deriv(s, u, p), np.zeros(4), 0.0)
    err = max(np.max(np.abs(Ac - Ac_n)), np.max(np.abs(Bc - Bc_n)))
    record("L1. analytic Ac,Bc == finite-diff Jacobian of deriv()",
           err < 1e-7, f"max abs diff {err:.3e}")

    Ad, Bd = discretise(Ac, Bc, p.tau)
    Ad_n, Bd_n = jacobian(lambda s, u: cp.euler_step(s, u, p), np.zeros(4), 0.0)
    err_d = max(np.max(np.abs(Ad - Ad_n)), np.max(np.abs(Bd - Bd_n)))
    record("L2. Ad = I + tau*Ac exact vs euler_step() Jacobian",
           err_d < 1e-7, f"max abs diff {err_d:.3e}")
    return Ac, Bc, Ad, Bd


def check_L3_L4_L5(Ad, Bd, Q, R):
    t0 = time.perf_counter()
    P, iters, delta = dare_iterate(Ad, Bd, Q, R)
    dt = time.perf_counter() - t0
    res = dare_residual(Ad, Bd, Q, R, P)
    record("L3. converged P satisfies the DARE", res < 1e-9,
           f"residual {res:.3e}, {iters} sweeps, last delta {delta:.1e}, {dt*1e3:.1f} ms")

    K = lqr_gain(Ad, Bd, Q, R, P)
    Acl = Ad - Bd @ K
    rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))
    record("L4. closed-loop spectral radius < 1", rho < 1.0, f"rho = {rho:.6f}")

    # Independent check on P: roll the LINEAR closed loop forward and accumulate
    # the quadratic stage cost. Must equal x0' P x0.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        x = rng.normal(size=4) * 0.05
        x0 = x.copy()
        cost = 0.0
        for _ in range(20_000):
            u = -K @ x
            cost += float(x @ Q @ x) + float(u @ R @ u)
            x = Acl @ x
            if np.max(np.abs(x)) < 1e-18:
                break
        pred = float(x0 @ P @ x0)
        worst = max(worst, abs(cost - pred) / max(abs(pred), 1e-30))
    record("L5. x'Px == simulated linear cost-to-go", worst < 1e-8,
           f"max rel err {worst:.3e}")
    return P, K, rho


def check_L6_cost_scaling(Ad, Bd, Q, R):
    """Only sign(-Kx) reaches the plant, so any positive rescaling of K is a
    no-op. Scaling the whole cost, (Q,R) -> (cQ,cR), scales P by c and leaves K
    exactly unchanged -- that is the invariance that actually holds."""
    P0, _, _ = dare_iterate(Ad, Bd, Q, R)
    K0 = lqr_gain(Ad, Bd, Q, R, P0)
    worst = 0.0
    for c in (1e-3, 1.0, 1e3):
        Pc, _, _ = dare_iterate(Ad, Bd, c * Q, c * R)
        Kc = lqr_gain(Ad, Bd, c * Q, c * R, Pc)
        worst = max(worst, float(np.max(np.abs(Kc - K0)) / np.max(np.abs(K0))))
    record("L6. (cQ, cR) leaves the gain unchanged", worst < 1e-9,
           f"max rel diff {worst:.3e} over c in [1e-3, 1e3]")

    # The claim I got wrong, kept as a measurement rather than deleted.
    ks, rets = [], []
    for r in (0.01, 1.0, 100.0):
        Pr, _, _ = dare_iterate(Ad, Bd, Q, np.array([[r]]))
        Kr = lqr_gain(Ad, Bd, Q, np.array([[r]]), Pr)
        ks.append(Kr.ravel() / np.linalg.norm(Kr))
        rets.append(float(evaluate(make_bangbang_policy(Kr), 100, seed=0).mean()))
    spread = max(float(np.max(np.abs(ks[0] - k))) for k in ks[1:])
    note("      R alone DOES rotate the switching surface",
         f"max direction diff {spread:.3f} for R in [0.01, 100]")
    note("      ...and moves the measured return by",
         f"{max(rets) - min(rets):.1f}  (all {rets[0]:.1f}/500) -- return is saturated")


# ------------------------------------------------------------------ basin ---

def rollout(K, s0, params=DEFAULT, max_steps=MAX_EPISODE_STEPS,
            mode="bangbang", clip=None):
    """Steps survived from a chosen initial state. Same plant, same termination
    test as CartPoleEnv, but with the initial state under our control."""
    s = np.asarray(s0, dtype=np.float64).copy()
    if abs(s[X]) > cp.X_THRESHOLD or abs(s[THETA]) > cp.THETA_THRESHOLD:
        return 0
    for t in range(max_steps):
        u = -(K @ s).item()
        if mode == "bangbang":
            f = params.force_mag if u > 0.0 else -params.force_mag
        else:
            f = u if clip is None else max(-clip, min(clip, u))
        s = cp.euler_step(s, f, params)
        if abs(s[X]) > cp.X_THRESHOLD or abs(s[THETA]) > cp.THETA_THRESHOLD:
            return t + 1
    return max_steps


def critical_magnitude(K, direction, mode, clip=None, tol=1e-4, cap=1e4):
    """Largest a for which a*direction survives 500 steps, by bisection.

    Survival need not be monotone in a, so this returns A boundary, not THE
    boundary. Good enough to say where the linear design stops working; not a
    proof of the region of attraction.
    """
    d = np.asarray(direction, dtype=np.float64)
    lo, hi = 0.0, 1.0
    while rollout(K, hi * d, mode=mode, clip=clip) >= MAX_EPISODE_STEPS:
        lo, hi = hi, hi * 2.0
        if hi > cap:
            return float("inf")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if rollout(K, mid * d, mode=mode, clip=clip) >= MAX_EPISODE_STEPS:
            lo = mid
        else:
            hi = mid
    return lo


def widened_init(policy, vel, episodes=200, seed=0):
    """Positions from the normal +-0.05 box, velocities from +-vel."""
    rng = np.random.default_rng(seed)
    rets = np.empty(episodes)
    for i in range(episodes):
        s0 = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-vel, vel),
                       rng.uniform(-0.05, 0.05), rng.uniform(-vel, vel)])
        rets[i] = rollout(policy, s0, mode="bangbang")
    return rets


# -------------------------------------------------------------------- main --

def main():
    p = DEFAULT
    print("\n=== step 2: LQR baseline ===\n")
    print("-- linearisation --")
    Ac, Bc, Ad, Bd = check_L1_L2(p)

    Q = np.eye(4)
    R = np.array([[1.0]])
    print("\n-- riccati (Q = I4, R = 1) --")
    P, K, rho = check_L3_L4_L5(Ad, Bd, Q, R)
    pred = math.log(1e-13) / math.log(rho ** 2)
    note("      sweeps predicted from the rate rho^2",
         f"~{pred:.0f} to reach 1e-13 relative (rho^2 = {rho**2:.5f})")

    print("\n-- design invariance --")
    check_L6_cost_scaling(Ad, Bd, Q, R)

    print("\n-- performance, sign-projected LQR (the baseline) --")
    policy = make_bangbang_policy(K)
    t0 = time.perf_counter()
    ret100 = evaluate(policy, episodes=100, seed=0)
    ret1000 = evaluate(policy, episodes=1000, seed=1)
    dt = time.perf_counter() - t0
    record("L7. mean return >= 475 over 100 consecutive episodes",
           ret100.mean() >= 475.0,
           f"{ret100.mean():.1f} (min {ret100.min():.0f}, max {ret100.max():.0f})")
    note("      over 1000 episodes",
         f"mean {ret1000.mean():.2f}, min {ret1000.min():.0f}, "
         f"failures {(ret1000 < 500).sum()}/1000")
    note("      sample cost of the design", "0 environment steps")
    note("      steps-to-threshold", "0 environment steps")
    note("      evaluation wall clock", f"{dt:.1f} s for 1100 episodes")

    print("\n-- where the linear design actually breaks (bisection) --")
    dirs = (("theta_dot0  [0,0,0,1]", [0., 0., 0., 1.], "rad/s"),
            ("x_dot0      [0,1,0,0]", [0., 1., 0., 0.], "m/s"),
            ("theta0      [0,0,1,0]", [0., 0., 1., 0.], "rad"))
    basin = {}
    for label, d, unit in dirs:
        cb = critical_magnitude(K, d, mode="bangbang")
        cc = critical_magnitude(K, d, mode="continuous", clip=p.force_mag)
        basin[label.split()[0]] = {"bangbang": cb, "continuous_clipped": cc}
        note(f"      critical {label}",
             f"bang-bang {cb:7.4f} {unit},  continuous(+-10N) {cc:7.4f} {unit}")
    note("      theta0 note",
         f"termination limit is {cp.THETA_THRESHOLD:.4f} rad, so any recoverable "
         f"angle is recoverable")

    print("\n-- robustness to a widened initial-velocity box --")
    for vel in (0.05, 0.5, 1.0, 1.5, 2.0, 2.5):
        r = widened_init(K, vel, episodes=200, seed=7)
        note(f"      velocities ~ U(-{vel}, {vel})",
             f"mean {r.mean():7.1f}, failures {(r < 500).sum():>3d}/200")

    print("\n-- Q sensitivity (R = 1 throughout) --")
    qsens = {}
    for label, Qc in (("I4                 ", np.eye(4)),
                      ("diag(1,1,10,10)    ", np.diag([1., 1., 10., 10.])),
                      ("diag(10,10,1,1)    ", np.diag([10., 10., 1., 1.])),
                      ("diag(0,0,1,1) pole ", np.diag([0., 0., 1., 1.]))):
        Pc, it, _ = dare_iterate(Ad, Bd, Qc, R)
        Kc = lqr_gain(Ad, Bd, Qc, R, Pc)
        rc = float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ Kc))))
        m = float(evaluate(make_bangbang_policy(Kc), episodes=100, seed=0).mean())
        crit = critical_magnitude(Kc, [0., 0., 0., 1.], mode="bangbang")
        # how far the cart wanders in 500 steps from the standard init
        env = CartPoleEnv(seed=3)
        obs, _ = env.reset()
        pol = make_bangbang_policy(Kc)
        maxx = 0.0
        while True:
            obs, _, term, trunc, _ = env.step(pol(obs))
            maxx = max(maxx, abs(obs[X]))
            if term or trunc:
                break
        qsens[label.strip()] = {"rho": rc, "mean_return": m,
                                "crit_theta_dot": crit, "max_abs_x": maxx}
        note(f"      Q = {label}",
             f"{it:>5d} sweeps, rho {rc:.4f}, return {m:5.1f}, "
             f"crit theta_dot {crit:5.2f} rad/s, max|x| {maxx:.3f} m")

    print("\n-- the numbers --")
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(f"  Ac =\n{Ac}\n")
    print(f"  Bc = {Bc.ravel()}")
    print(f"  K  = {K.ravel()}   (u = -Kx, action = 1 if u > 0)")
    print(f"  P  =\n{P}\n")

    out = {
        "Ac": Ac.tolist(), "Bc": Bc.ravel().tolist(),
        "Ad": Ad.tolist(), "Bd": Bd.ravel().tolist(),
        "Q": Q.tolist(), "R": R.tolist(), "P": P.tolist(), "K": K.ravel().tolist(),
        "closed_loop_spectral_radius": rho,
        "mean_return_100": float(ret100.mean()),
        "mean_return_1000": float(ret1000.mean()),
        "sample_cost_env_steps": 0,
        "steps_to_threshold": 0,
        "basin": basin,
        "q_sensitivity": qsens,
    }
    with open("lqr_baseline.json", "w") as f:
        json.dump(out, f, indent=2)

    ok = all(_RESULTS)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed"
          f"{'' if ok else '  <-- FAILURES'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
