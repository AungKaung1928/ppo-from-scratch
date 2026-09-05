# ppo-from-scratch — PPO, measured honestly against a closed-form controller

Cart-pole. A hand-written environment, an LQR baseline that costs zero samples,
and PPO implemented from scratch. The question the repo answers is not "can PPO
balance a pole" — it can — but **what does the learned policy actually buy over a
controller you can solve for in closed form, and what does it cost in samples.**

No gymnasium. No stable-baselines3. No rllib. No cleanrl copy-paste. No scipy —
the discrete Riccati equation is solved by iterating the recursion, because
understanding that recursion is the point.

Secondary project. `../mujoco-clutter-detect` has priority; nothing here runs
while that one is training.

## Status

| step | what | state |
|---|---|---|
| 1 | Hand-written CartPole env, verified against the published equations | **done, 13/13 checks pass** |
| 2 | LQR baseline: linearise, iterate the Riccati recursion, report zero sample cost | next |
| 3 | PPO from scratch, single file, categorical policy | |
| 4 | Seed study: >=5 seeds, median + IQR, steps-to-threshold | |
| 5 | Ablations: no GAE, no advantage normalisation, no ratio clipping | |
| 6 | *stretch* continuous-action MuJoCo version, Gaussian policy | |

**Solved** = mean return >= 475 over 100 consecutive episodes, episode capped at
500 steps. **Steps-to-threshold** = environment steps consumed before that is
first reached. Both are the CartPole-v1 convention, on purpose, so the numbers
here can be read against anyone else's.

---

## Step 1 — the environment

`cartpole.py`, 209 lines, numpy only. `python verify_env.py` runs every check
below in about 6 s on one core.

### Why hand-write it

PPO will fail to learn several times before step 5 is finished. Every time it
does, the first question is *algorithm or environment*. That question is only
cheap to answer if the environment was verified before any RL existed. This one
was.

### The dynamics

Barto, Sutton & Anderson (1983), IEEE SMC-13(5), appendix. Theta from upright,
positive theta leans the pole toward +x.

```
temp      = (F + m l thetadot^2 sin th) / (M + m)
thetaddot = (g sin th - cos th * temp) / (l (4/3 - m cos^2 th / (M + m)))
xddot     = temp - m l thetaddot cos th / (M + m)
```

`l = 0.5` is the **half**-length. The `4/3` is `1 + I_cm/(m l^2)` for a uniform
rod, whose moment of inertia about its own centre is `m (2l)^2 / 12 = m l^2 / 3`.
Getting the half-length convention wrong is the most common bug in a hand-written
cart-pole, so check 3 exists purely to catch it.

### Verification — 13 checks, all passing

| # | check | result |
|---|---|---|
| 1 | Scalar formulas vs an independent solve of the Lagrangian mass matrix, 10k random states, `abs(theta)` up to pi | max rel diff **2.6e-15** |
| 2a | Energy conserved under RK4, F=0, 2 s at dt=1e-4 | max rel drift **2.4e-14** |
| 2b | Horizontal momentum conserved, same run | max abs drift **5.4e-15** |
| 3 | Pinned cart, divergence rate vs closed form `sqrt(3g/4l)` | measured **3.834058** vs **3.834058** s⁻¹, rel err **3.4e-11** |
| 4a | Truncation fires at exactly the cap, `terminated` stays `False` | exact |
| 4b | Stepping a finished episode raises | `RuntimeError` |
| 4c | Reward is 1.0 on every step including the terminal one | always-right policy: 9 steps, return 9.0 |
| 4d | Termination bounds are `abs(x)>2.4` and `abs(theta)>12°` | 0.20943951023931953 rad |
| 4e | Action sign: bang-bang `theta + 0.5 thetadot` balances | **500.0 / 500** over 100 episodes |
| 5a | Same seed → bitwise identical trajectory; different seed differs | exact |
| 5b | Env ignores the global numpy RNG | unchanged under `np.random.seed` |
| 6 | Random-policy mean return vs the published ~22 | **22.09 ± 0.12** over 10k episodes |
| 7 | Golden trajectory regression | bitwise identical, 201 states |

Check 1 is the strong one: the mass matrix

```
[ M+m         m l cos th ] [ xddot  ]   [ F + m l thetadot^2 sin th ]
[ m l cos th  4/3 m l^2  ] [ thddot ] = [ m g l sin th              ]
```

is assembled from the Lagrangian and inverted with `np.linalg.solve`. It shares
no algebra with the hand-eliminated formulas the env uses, so agreement to 2.6e-15
rules out sign errors and bad elimination. Check 3 is the one that reaches
outside the code entirely: `sqrt(3g/4l) = 3.834058 s⁻¹` is a number you can
derive on paper, and the simulator reproduces it to 11 digits.

**What none of this proves.** gymnasium is not installed, so there is no bit-diff
against the reference implementation. Checks 1–3 prove the equations are the
published ones and are internally consistent; check 6 is the only externally
published number reproduced. Check 7 is a regression fixture, not a correctness
proof — it freezes today's behaviour so a refactor during steps 3–5 cannot move
the env silently.

### Measured properties of the plant

| quantity | value |
|---|---|
| Instability rate about upright, pinned cart | **3.834 s⁻¹** (e-fold in 261 ms) |
| One-step Euler error vs RK4, in the operational envelope | median **2.9e-3 rad**, max 3.5e-3 rad (1.7% of the 12° limit) |
| Euler vs RK4 after 25 controlled steps (0.5 s), identical forces | **7.8e-3 rad** (3.7% of the limit) |
| Explicit-Euler energy gain, F=0, 50 steps | **+0.43%** |
| Scalar env throughput, one core | **~374,000 steps/s** |

### Three decisions worth arguing with

**The integrator is explicit Euler, and that is deliberate.** Position is
advanced with the *old* velocity — v1's default. Semi-implicit Euler is the
better integrator and is not used, because the deliverable is a number
("475/500") that is only comparable to published results if the plant is the same
plant. The cost is measured above and not hidden: the v1 simulator is a
materially different system from the true ODE, roughly 3.7% of the failure angle
after half a second of control.

A full-episode Euler-vs-RK4 comparison is **not** reported, and the reason is
check 3: at `lambda = 3.834 s⁻¹`, any difference between two integrators is
amplified by `e^38 ≈ 3e16` over a 10 s episode. That number would be about
Lyapunov exponents, not about integrators. An early draft of the harness reported
it anyway and got `max abs(d theta) = 12.9 rad`, which is how the mistake was caught.

**`terminated` and `truncated` are separate and there is no combined `done`.**
Collapsing them makes the value target at the 500-step cap treat a perfectly
healthy upright state as worth zero future return. That is a real bug worth
roughly 100 points of final performance, and the env refuses to expose the flag
that enables it.

**The vectorised env planned for step 3 is probably not needed.** The plan
assumed the python-loop env would be the bottleneck. Measured: 374k steps/s, so a
200k-step run spends **0.5 s** total inside the environment — about 1% of the
10-minute budget. torch forward/backward will dominate. It gets written only if a
profile says so.

### Deviations from CartPole-v1, complete

1. Observations are float64, not float32.
2. Stepping a finished episode raises instead of warning and returning reward 0.
3. `terminated` / `truncated` are never merged.

Everything else — constants, equations, integrator and its ordering, init
distribution, thresholds, reward, 500-step cap — matches.

## Running

```bash
source ~/personal/ml/env.sh
python verify_env.py          # ~6 s, one core, prints the table above
```

No dependencies beyond numpy, already in the shared `~/personal/ml/.venv`.
Nothing in this repo installs anything: the venv is shared with the detection
project and an install here changes that project's environment mid-run.
