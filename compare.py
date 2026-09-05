"""
LQR vs PPO, on the axis where return cannot tell them apart.

Both controllers score 500.0/500 on the standard task. Step 2 already showed
that return saturates and therefore cannot rank anything competent, so ranking
them by return would be reporting a tie that means nothing.

What can be measured instead:
  - sample cost of the design (LQR: 0. PPO: the median steps-to-threshold.)
  - the basin: how large an initial disturbance each one recovers from.

The second is the one that matters for anything that leaves a simulator. A
policy that holds the nominal task but has a small basin is a policy that falls
over the first time the real plant hands it a disturbance the training
distribution never contained -- and the CartPole-v1 init range, +-0.05 on every
state, is a very small disturbance. Step 2 measured that the LQR does not even
get probed by it.

This file runs both controllers through the same widened-initial-condition
sweep and the same one-axis bisection, using the same plant and the same
termination test.
"""

import glob
import json
import os

import numpy as np
import torch

import cartpole as cp
import lqr
from cartpole import MAX_EPISODE_STEPS, THETA, X, CartPoleEnv
from ppo import load_agent


def ppo_policy_fn(agent):
    """Greedy: argmax over logits. The stochastic policy is what training used;
    the deployed thing is the argmax, so that is what gets measured."""
    def act(s):
        with torch.no_grad():
            return int(torch.argmax(agent.actor(torch.as_tensor(s, dtype=torch.float32))))
    return act


def rollout_policy(act, s0, params=cp.DEFAULT, max_steps=MAX_EPISODE_STEPS):
    s = np.asarray(s0, dtype=np.float64).copy()
    if abs(s[X]) > cp.X_THRESHOLD or abs(s[THETA]) > cp.THETA_THRESHOLD:
        return 0
    for t in range(max_steps):
        f = params.force_mag if act(s) == 1 else -params.force_mag
        s = cp.euler_step(s, f, params)
        if abs(s[X]) > cp.X_THRESHOLD or abs(s[THETA]) > cp.THETA_THRESHOLD:
            return t + 1
    return max_steps


def critical(act, direction, tol=1e-4, cap=1e4):
    d = np.asarray(direction, dtype=np.float64)
    lo, hi = 0.0, 1.0
    while rollout_policy(act, hi * d) >= MAX_EPISODE_STEPS:
        lo, hi = hi, hi * 2.0
        if hi > cap:
            return float("inf")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if rollout_policy(act, mid * d) >= MAX_EPISODE_STEPS:
            lo = mid
        else:
            hi = mid
    return lo


def widened(act, vel, episodes=200, seed=7):
    rng = np.random.default_rng(seed)
    r = np.empty(episodes)
    for i in range(episodes):
        r[i] = rollout_policy(act, [rng.uniform(-0.05, 0.05), rng.uniform(-vel, vel),
                                    rng.uniform(-0.05, 0.05), rng.uniform(-vel, vel)])
    return r


def main():
    Ac, Bc = lqr.linearise_analytic()
    Ad, Bd = lqr.discretise(Ac, Bc, cp.DEFAULT.tau)
    Q, R = np.eye(4), np.array([[1.0]])
    P, _, _ = lqr.dare_iterate(Ad, Bd, Q, R)
    K = lqr.lqr_gain(Ad, Bd, Q, R, P)
    lqr_act = lqr.make_bangbang_policy(K)

    ckpts = sorted(glob.glob(os.path.join("runs", "study", "baseline_seed*.pt")))
    if not ckpts:
        raise SystemExit("no trained baseline checkpoints in runs/study -- "
                         "run `python study.py --mode ablations` first")
    agents = [(os.path.basename(c).replace(".pt", ""), ppo_policy_fn(load_agent(c)[0]))
              for c in ckpts]
    print(f"  {len(agents)} PPO checkpoints, 1 LQR gain\n")

    print("-- nominal task (standard +-0.05 init, 100 episodes) --")
    def nominal(act, seed=0):
        env = CartPoleEnv(seed=seed)
        out = []
        for _ in range(100):
            obs, _ = env.reset()
            tot = 0.0
            while True:
                obs, r, term, trunc, _ = env.step(act(obs))
                tot += r
                if term or trunc:
                    break
            out.append(tot)
        return float(np.mean(out))
    lqr_nom = nominal(lqr_act)
    ppo_nom = np.array([nominal(a) for _, a in agents])
    print(f"  LQR  mean return {lqr_nom:.1f}")
    print(f"  PPO  mean return median {np.median(ppo_nom):.1f}, "
          f"min over seeds {ppo_nom.min():.1f}, "
          f"IQR [{np.percentile(ppo_nom,25):.1f}, {np.percentile(ppo_nom,75):.1f}]")
    print("  -> tie. This is the measurement that cannot distinguish them.\n")

    print("-- critical single-axis disturbance (bisection) --")
    axes = (("theta_dot0", [0., 0., 0., 1.], "rad/s"), ("x_dot0", [0., 1., 0., 0.], "m/s"))
    table = {}
    for name, d, unit in axes:
        l = critical(lqr_act, d)
        p = np.array([critical(a, d) for _, a in agents])
        table[name] = {"lqr": l, "ppo": p.tolist()}
        print(f"  {name:<11} LQR {l:6.3f} {unit:<6} | "
              f"PPO median {np.median(p):6.3f}  IQR [{np.percentile(p,25):.3f}, "
              f"{np.percentile(p,75):.3f}]  min {p.min():.3f}  max {p.max():.3f}")

    print("\n-- widened initial-velocity box, failures out of 200 --")
    print(f"  {'velocities':<20} {'LQR':>8} {'PPO median':>12} {'PPO worst seed':>16}")
    wide = {}
    for vel in (0.5, 1.0, 1.5, 2.0, 2.5):
        l = (widened(lqr_act, vel) < MAX_EPISODE_STEPS).sum()
        p = np.array([(widened(a, vel) < MAX_EPISODE_STEPS).sum() for _, a in agents])
        wide[vel] = {"lqr": int(l), "ppo": p.tolist()}
        print(f"  U(-{vel}, {vel}){'':<10} {l:>8d} {np.median(p):>12.1f} {p.max():>16d}")

    with open(os.path.join("runs", "compare.json"), "w") as f:
        json.dump({"nominal_lqr": lqr_nom, "nominal_ppo": ppo_nom.tolist(),
                   "critical": table, "widened": wide}, f, indent=2)
    print("\n  wrote runs/compare.json\n")


if __name__ == "__main__":
    main()
