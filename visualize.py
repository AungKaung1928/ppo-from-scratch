"""
Watch the cart-pole in the MuJoCo viewer.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
MuJoCo here is a RENDERER, not the plant. Every number in this repo -- the 13
env checks, the LQR gain, the 16-seed study, the ablations -- is measured
against the numpy dynamics in cartpole.py. This script integrates that same
numpy plant and writes the resulting (x, theta) straight into MuJoCo's qpos,
then calls mj_forward for kinematics only. mj_step is never called.

Doing it the other way round -- letting MuJoCo integrate and calling that "the
same environment" -- would silently change the plant. Different integrator,
different joint handling, and cartpole.py's deliberate bit-compatibility with
CartPole-v1's explicit Euler would be gone. The model file matches the physical
parameters so that a future step 6 CAN hand the dynamics to MuJoCo, but that
would be a different environment and would need its own baseline.

On this box MuJoCo renders through llvmpipe (software GL), so expect a
low frame rate on a full-size window. It is a 2-body scene; it is fine.

USAGE
    source ~/personal/ml/env.sh
    python visualize.py --controller lqr
    python visualize.py --controller ppo --ckpt runs/study/baseline_seed0.pt
    python visualize.py --controller lqr --theta-dot0 2.0     # near the basin edge
    python visualize.py --controller lqr --record out/frames  # no window needed
"""

import argparse
import os
import time

import numpy as np

import cartpole as cp
from cartpole import MAX_EPISODE_STEPS, THETA, THETA_DOT, X, X_DOT


def build_controller(kind, ckpt):
    if kind == "random":
        rng = np.random.default_rng(0)
        return lambda s: int(rng.integers(2)), "random policy"
    if kind == "lqr":
        import lqr as L
        Ac, Bc = L.linearise_analytic()
        Ad, Bd = L.discretise(Ac, Bc, cp.DEFAULT.tau)
        Q, R = np.eye(4), np.array([[1.0]])
        P, _, _ = L.dare_iterate(Ad, Bd, Q, R)
        K = L.lqr_gain(Ad, Bd, Q, R, P)
        return L.make_bangbang_policy(K), f"LQR bang-bang, K = {np.round(K.ravel(), 3)}"
    if kind == "ppo":
        import torch
        from ppo import load_agent
        agent, c = load_agent(ckpt)

        def act(s):
            with torch.no_grad():
                return int(torch.argmax(agent.actor(torch.as_tensor(s, dtype=torch.float32))))
        return act, f"PPO greedy, {os.path.basename(ckpt)}"
    raise ValueError(kind)


def simulate(act, s0, max_steps):
    """Roll the numpy plant. Returns the state history and why it ended."""
    s = np.asarray(s0, dtype=np.float64).copy()
    hist = [s.copy()]
    for t in range(max_steps):
        f = cp.DEFAULT.force_mag if act(s) == 1 else -cp.DEFAULT.force_mag
        s = cp.euler_step(s, f, cp.DEFAULT)
        hist.append(s.copy())
        if abs(s[X]) > cp.X_THRESHOLD:
            return np.array(hist), f"cart left the rail at step {t+1}"
        if abs(s[THETA]) > cp.THETA_THRESHOLD:
            return np.array(hist), f"pole exceeded 12 deg at step {t+1}"
    return np.array(hist), f"survived the full {max_steps} steps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["lqr", "ppo", "random"], default="lqr")
    ap.add_argument("--ckpt", default="runs/study/baseline_seed0.pt")
    ap.add_argument("--steps", type=int, default=MAX_EPISODE_STEPS)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta0", type=float, default=None)
    ap.add_argument("--theta-dot0", type=float, default=None)
    ap.add_argument("--x-dot0", type=float, default=None)
    ap.add_argument("--zero-init", action="store_true",
                    help="start from exactly [0,0,0,0] before applying the "
                         "--theta0/--theta-dot0/--x-dot0 overrides, so a single "
                         "axis is isolated. Without it the random +-0.05 init "
                         "is still there and can help or hurt -- a negative "
                         "theta0 with a positive theta_dot0 is the pole "
                         "rotating BACK towards upright, which is easier, not "
                         "harder. The basin numbers in the README are all "
                         "bisected from the origin, so use this to reproduce "
                         "them.")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="render offscreen to PNG frames instead of opening a window")
    ap.add_argument("--every", type=int, default=4, help="record every Nth step")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    s0 = (np.zeros(4) if a.zero_init
          else rng.uniform(-cp.INIT_RANGE, cp.INIT_RANGE, size=4))
    for key, i in (("theta0", THETA), ("theta_dot0", THETA_DOT), ("x_dot0", X_DOT)):
        v = getattr(a, key)
        if v is not None:
            s0[i] = v

    act, label = build_controller(a.controller, a.ckpt)
    hist, why = simulate(act, s0, a.steps)
    print(f"  controller : {label}")
    print(f"  s0         : {np.round(s0, 5)}")
    print(f"  outcome    : {why}  ({len(hist)-1} steps, "
          f"{(len(hist)-1)*cp.DEFAULT.tau:.2f} s simulated)")
    print(f"  max |theta|: {np.abs(hist[:, THETA]).max():.4f} rad "
          f"({np.abs(hist[:, THETA]).max()/cp.THETA_THRESHOLD:.0%} of the limit)")
    print(f"  max |x|    : {np.abs(hist[:, X]).max():.4f} m")

    import mujoco
    model = mujoco.MjModel.from_xml_path("cartpole.xml")
    data = mujoco.MjData(model)

    def place(state):
        data.qpos[0] = state[X]
        data.qpos[1] = state[THETA]
        mujoco.mj_forward(model, data)

    if a.record:
        os.makedirs(a.record, exist_ok=True)
        # matplotlib, not imageio: imageio is not in the shared venv and this
        # repo does not install anything into it.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        n = 0
        with mujoco.Renderer(model, 640, 960) as r:
            for k in range(0, len(hist), a.every):
                place(hist[k])
                r.update_scene(data, camera="side")
                mpimg.imsave(os.path.join(a.record, f"f{k:05d}.png"), r.render())
                n += 1
        print(f"  wrote {n} frames to {a.record}/  "
              f"(stitch with: ffmpeg -framerate 25 -pattern_type glob "
              f"-i '{a.record}/*.png' out.mp4)")
        return

    import mujoco.viewer
    print("\n  opening the MuJoCo viewer (software GL on this box -- expect a "
          "modest frame rate).\n  close the window to exit.\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -12.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.4]
        dt = cp.DEFAULT.tau / max(a.speed, 1e-6)
        while viewer.is_running():
            for state in hist:
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()
                place(state)
                viewer.sync()
                slack = dt - (time.perf_counter() - t0)
                if slack > 0:
                    time.sleep(slack)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
