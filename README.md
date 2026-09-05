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
| 2 | LQR baseline: linearise, iterate the Riccati recursion, report zero sample cost | **done, 7/7 checks pass** |
| 3 | PPO from scratch, single file, categorical policy | next |
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

---

## Step 2 — the LQR baseline

`lqr.py`. Runs in ~30 s on one core. 7/7 checks pass. Numbers land in
`lqr_baseline.json`.

The point of this step is to fix the number PPO has to justify itself against.
Not "random", not a hand-tuned heuristic — the optimal linear controller for the
linearised plant, designed from the model alone, at a sample cost of **zero
environment steps**.

### Linearisation

Dropping second-order terms in `cartpole.accelerations()` about `(0,0,0,0)`,
with `D = l(4/3 - m/M)`:

```
thetaddot = (g/D) theta - (1/(M D)) F
xddot     = F/M - (m l/M) thetaddot
```

which gives, numerically:

```
Ac = [[0, 1,  0      , 0],        Bc = [0, 0.975610, 0, -1.463415]'
      [0, 0, -0.717073, 0],
      [0, 0,  0      , 1],
      [0, 0, 15.775610, 0]]
```

`Ac[3,2] = g/D = 15.7756 s^-2`, and `sqrt(15.7756) = 3.9719 s^-1` is the growth
rate of the *free* cart-pole — slightly faster than the pinned-cart
`3.834058 s^-1` measured in step 1, because a free cart lets the base slide away
under the pole.

**Discretisation is exact here, and that is the payoff for the integrator we
kept in step 1.** Every acceleration in `euler_step()` is evaluated at the old
state, so the plant genuinely is `s' = s + tau f(s,u)` and therefore

```
Ad = I + tau*Ac      Bd = tau*Bc
```

with no approximation at all. A semi-implicit or exact-ZOH plant would need a
matrix exponential here. Verified against a finite-difference Jacobian of
`euler_step()` anyway: max abs diff **7.57e-14**.

### Riccati by iteration, not by library call

```
P_{k+1} = Q + Ad' P_k Ad - Ad' P_k Bd (R + Bd' P_k Bd)^-1 Bd' P_k Ad,   P_0 = Q
K       = (R + Bd' P Bd)^-1 Bd' P Ad,    u = -Kx
```

`P_k` is the cost-to-go matrix with `k` steps of horizon left, so this is
literally value iteration for a quadratic value function. It is the same object
PPO's critic estimates by regression in step 3 — the difference is that here it
can be computed exactly, which is the whole reason to write the recursion out.

Converged in **866 sweeps** / 22 ms. The theory says the rate is
`rho(Ad-BdK)^2 = 0.96810` per sweep, predicting **~923** sweeps to reach 1e-13
relative. Measured 866. That agreement is the check that the recursion is the
one in the textbook and not something that merely happens to converge.

```
K = [-0.910126, -2.132488, -30.594563, -7.841506]      (u = -Kx)
rho(Ad - Bd K) = 0.983919
```

### Checks

| # | check | result |
|---|---|---|
| L1 | analytic `Ac,Bc` == central-difference Jacobian of `deriv()` | max abs **3.78e-12** |
| L2 | `Ad = I + tau Ac` == finite-difference Jacobian of `euler_step()` | max abs **7.57e-14** |
| L3 | converged `P` satisfies the DARE | residual **6.63e-10**, 866 sweeps |
| L4 | closed-loop spectral radius < 1 | **0.983919** |
| L5 | `x'Px` == simulated linear closed-loop cost-to-go | max rel **5.46e-12** |
| L6 | `(cQ, cR)` leaves the gain unchanged | max rel **5.81e-16** over `c` in [1e-3, 1e3] |
| L7 | mean return >= 475 over 100 consecutive episodes | **500.0** (min 500, max 500) |

L5 is the one that matters most: it checks `P` against something `P` was not
used to compute — roll the *linear* closed loop forward, accumulate
`x'Qx + u'Ru`, and compare the total to `x0' P x0`. Agreement to 5e-12 means the
converged matrix really is the cost-to-go and not just a fixed point of some
recursion I typed.

### Two actions, one continuous controller

The env has two actions, ±10 N. LQR returns a real `u`. The bridge is a design
decision, not a detail:

```
action = 1 if -Kx > 0 else 0
```

Bang-bang with an LQR-derived switching surface. Only the *sign* of `-Kx`
reaches the plant, so the policy is invariant to positive rescaling of `K`, and
therefore to scaling the whole cost `(Q,R) -> (cQ,cR)` — that is L6.

### The baseline number

| metric | sign-projected LQR |
|---|---|
| mean return, 100 consecutive episodes | **500.0** |
| mean return, 1000 episodes | **500.00**, 0 failures |
| sample cost of the design | **0 environment steps** |
| steps-to-threshold | **0** |

**PPO cannot beat this.** The metric is saturated. The most it can do is match
it, and the only honest axes left are sample cost — where LQR wins by
construction, since it consumed none — and everything the LQR does not have,
which is a model. That framing is the deliverable of this step, and it is what
step 4 and step 5 are measured inside.

### Return is a saturating metric — measured, not asserted

I originally wrote that `R` alone cannot matter because it only rescales `K`.
That is false — the DARE is not linear in `R`, and sweeping `R` from 0.01 to 100
rotates the normalised gain direction by **0.130**. The check caught it. The
interesting part is what happened next: that rotation moved the measured return
by **0.0**. Every one of those controllers scores 500.0/500.

So return cannot rank controllers on this task once they are all merely good
enough. Steps 4 and 5 therefore rank on steps-to-threshold, and this repo does
not report a single reward curve anywhere.

What *can* tell these controllers apart is the basin, found by bisecting on
initial state along one axis at a time:

| perturbation from upright | bang-bang (±10 N only) | continuous force, clipped to ±10 N |
|---|---|---|
| critical `theta_dot0` | **2.1693 rad/s** | 1.9632 rad/s |
| critical `x_dot0` | **2.4354 m/s** | 1.9593 m/s |
| critical `theta0` | 0.2094 rad = the termination limit | 0.2094 rad |

Bang-bang has the *larger* basin than the same LQR law with a real actuator of
the same strength. That is not a paradox: recovering from a large disturbance is
a time-optimal problem, and the time-optimal solution to a bounded-input problem
is bang-bang. The linear law under-commands near the edge because it is
minimising a quadratic cost, not maximising survival. Angle finds nothing —
every angle inside the termination limit is recoverable, which is why the
basin sweep is run on velocity.

Widening the initial-velocity box (positions stay at the standard ±0.05):

| initial velocities | mean return | failures / 200 |
|---|---|---|
| U(-0.05, 0.05) (the standard env) | 500.0 | 0 |
| U(-1.0, 1.0) | 500.0 | 0 |
| U(-1.5, 1.5) | 487.0 | 6 |
| U(-2.0, 2.0) | 436.9 | 28 |
| U(-2.5, 2.5) | 357.5 | 61 |

The standard env sits so far inside the basin that it never probes the
controller at all. Worth remembering when PPO scores 500 in step 4.

### Q sensitivity

`R = 1` throughout; only relative weights in `Q` do anything.

| Q | sweeps | rho | return | critical `theta_dot0` | max abs x over an episode |
|---|---|---|---|---|---|
| `I4` | 866 | 0.9839 | 500.0 | 2.17 rad/s | 0.136 m |
| `diag(1,1,10,10)` pole-weighted | 887 | 0.9842 | 500.0 | 2.14 rad/s | 0.143 m |
| `diag(10,10,1,1)` cart-weighted | 664 | 0.9786 | 500.0 | 1.89 rad/s | 0.081 m |
| `diag(0,0,1,1)` pole only | 240 | **1.0000** | 500.0 | **0.40 rad/s** | 0.245 m |

The last row is the interesting one. With no cost on cart position or velocity,
the cart mode is uncontrolled: `rho = 1.0000` exactly, the closed loop is only
marginally stable, and the cart drifts. It still scores 500.0, because 500 steps
is not long enough to drift 2.4 m. Return says these four designs are identical.
The spectral radius and the basin say the fourth is broken.

### A bug worth writing down

The Riccati loop first used an **absolute** convergence tolerance of 1e-14.
`P` has entries near 7e3, so `eps * 7e3 ~ 1.5e-12` — the tolerance was below the
float64 resolution of `P` itself and could only ever be met by luck. `Q = I4`
landed on an exact fixed point and converged; `Q = diag(10,10,1,1)` chattered in
the last bit for 200,000 sweeps and looked exactly like a stabilisability
failure. It was not. Fixed by making the tolerance relative to `||P||_inf`;
`diag(10,10,1,1)` now converges in 664 sweeps.

The general lesson, and it will come back in step 3: an absolute tolerance on a
quantity whose scale you have not measured is not a convergence criterion.

## Running

```bash
source ~/personal/ml/env.sh
python verify_env.py          # step 1, ~6 s, one core
python lqr.py                 # step 2, ~30 s, one core
```

No dependencies beyond numpy, already in the shared `~/personal/ml/.venv`.
Nothing in this repo installs anything: the venv is shared with the detection
project and an install here changes that project's environment mid-run.
