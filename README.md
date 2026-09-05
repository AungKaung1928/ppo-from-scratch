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
| 3 | PPO from scratch, single file, categorical policy | **done** |
| 4 | Seed study: >=5 seeds, median + IQR, steps-to-threshold | **done, 16 seeds** |
| 5 | Ablations: no GAE, no advantage normalisation, no ratio clipping | **done, 16 seeds each** |
| 6 | *stretch* continuous-action MuJoCo version, Gaussian policy | **done, 6/6 env checks, 16 seeds + 3 ablations** |

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

---

## Step 3 — PPO, written out

`ppo.py`, 358 lines. Nothing imported from stable-baselines3, rllib or cleanrl.
The four things that are actually PPO — GAE, advantage normalisation, the
clipped surrogate, and the multi-epoch minibatch update — are written longhand
and each one can be switched off from the command line, which is what step 5
measures.

Separate 64-64 tanh actor and critic trunks, orthogonal init (gain `sqrt(2)`,
`0.01` on the policy head, `1.0` on the value head), Adam with `eps=1e-5`,
linear LR anneal, gradient-norm clip at 0.5, 8 envs x 128 steps = 1024-step
batches.

### The truncation bootstrap — what step 1 was for

At the 500-step cap the pole has **not** fallen. The episode was cut by the
clock. Bootstrapping `V(s_final)` there is correct; treating it as a terminal
state teaches the critic that surviving 500 steps is worth zero future return.
That is the single most common silent bug in a hand-written PPO, and it is only
avoidable because step 1 refused to merge `terminated` and `truncated` into one
`done` flag. Implemented by folding the bootstrap into the reward of the
truncated step, `r_t += gamma * V(s_final)`, and then treating the step as an
episode boundary for GAE — which it is, since the next observation comes from a
fresh reset.

### Hyperparameters were searched, on seeds that are not the reported ones

The first configuration written down (`lr 3e-4`, 4 epochs) **never reached
threshold** in 150k steps, and reported `clipfrac 0.000` on every single update:
the policy was moving so little per update that the clip never engaged. That is
worth stating rather than quietly deleting, because a PPO whose clip never
engages is not PPO, it is A2C with extra arithmetic — and it would have made the
step-5 clipping ablation come out as "makes no difference".

Search on seeds **100–103**, disjoint from the study seeds 0–15:

| config | solved | steps-to-threshold, median | IQR |
|---|---|---|---|
| **`lr 2e-3`, 10 epochs — adopted** | **4/4** | **63,028** | [62,020, 63,428] |
| `lr 1e-3`, 8 epochs, 8 minibatches | 4/4 | 66,520 | [65,076, 69,772] |
| `lr 1.5e-3`, 10 epochs | 4/4 | 73,412 | [68,910, 78,664] |
| `lr 1e-3`, 10 epochs | 4/4 | 76,104 | [73,346, 85,050] |
| `lr 2.5e-3`, 4 epochs | 3/4 | 112,104 | [83,832, 141,888] |
| `lr 3e-4`, 10 epochs | 2/4 | 148,808 | — |
| `lr 1e-3`, 4 epochs | 1/4 | >150,000 | — |
| `lr 3e-4`, 4 epochs (first attempt) | 0/4 | >150,000 | — |

With the adopted config the clip fraction runs at 0.03–0.05 instead of 0.000.

### Cost

| | |
|---|---|
| one 150k-step run, sequential | **11.4 s** (13,187 env steps/s) |
| 64 runs (4 configs x 16 seeds), 4-way parallel, nice 10 | **344 s** |
| worst single run | ~21 s |
| budget | 10 minutes per run |

`torch.set_num_threads(1)`, deliberately. The nets are 64x64; intra-op threading
costs more in synchronisation than it saves, and this box shares 8 threads with a
detection-training job that has priority. `study.py` refuses to launch if the
1-minute load average says something else is already using the machine.

Step 1 predicted the env would not be the bottleneck (374k env steps/s standalone
vs 13k inside training). Confirmed: the env is ~3% of the wall clock. The
vectorised env rewrite was never needed and was never written.

---

## Step 4 — seed study

16 seeds (0–15), median and interquartile range. **No single reward curve
appears in this repo**, and `ppo.py` does not plot one.

| | baseline PPO | LQR (step 2) |
|---|---|---|
| solved (>=475 over 100 consecutive episodes) | **16/16** | yes |
| steps-to-threshold, median | **62,144** | **0** |
| steps-to-threshold, IQR | [60,442, 63,448] | — |
| greedy eval, 100 episodes, median | **500.0** | 500.0 |
| greedy eval, worst seed | 500.0 | — |

Reproducible: re-running the 16 baseline seeds a second time returned the
identical median and IQR to the step.

**PPO matches the baseline and pays 62,144 environment steps for what the model
gave away free.** That is the honest headline. It is not an argument against
PPO — it is a precise statement of what PPO is buying, which is independence
from the model, and of what that costs on a plant where the model happens to be
exact.

### Censoring

A run that never reaches threshold has no steps-to-threshold. Dropping those and
taking the median of the survivors is the standard way to make a bad
configuration look good, so they are entered as `budget + 1` instead. The median
is then correct whenever fewer than half the runs fail, and reports as
`>150000` when more than half do. Solved counts are printed next to every
median.

---

## Step 5 — ablations

16 seeds each, same budget, same seeds, one switch changed at a time.

| ablation | solved | steps-to-threshold median | vs baseline | permutation p |
|---|---|---|---|---|
| baseline | 16/16 | 62,144 | 1.00x | — |
| no advantage normalisation | 16/16 | 64,884 | 1.04x | **0.054** |
| no GAE (lambda = 1) | 16/16 | 68,948 | 1.11x | 0.0006 |
| **no ratio clipping** | **8/16** | **145,368** | **2.34x** | 0.0001 |

**Ratio clipping is the only one that actually matters, and it matters by a
lot.** Half the seeds never reach threshold without it. The other two are
measurable but small: GAE is worth 11% of the sample budget here, and advantage
normalisation is not distinguishable from noise at the 5% level on this task.

That ordering is task-specific and the reason is visible in the setup. Cart-pole
episodes are short and `gamma = 0.99` over 128-step rollouts, so `lambda = 1`
costs much less variance than it would on a long-horizon task. Clipping, by
contrast, is what makes 10 epochs on the same batch legal at all — without it,
the objective is a correct policy gradient for the first epoch and an
increasingly wrong one for the next nine.

### "no GAE" means lambda = 1, not "no advantage"

GAE(gamma, 1) *is* the plain Monte-Carlo advantage. Implementing the ablation as
`lambda = 1.0` changes exactly one number and nothing else in the code path,
which is the only way the comparison is about GAE rather than about two
different implementations.

### Significance, without scipy

Steps-to-threshold is right-skewed and censored, so a t-test is the wrong tool.
`study.py` runs a two-sided permutation test on the difference of medians
(exhaustive when the number of splits allows, 100,000 Monte-Carlo resamples
otherwise). It assumes only exchangeability under the null, which is exactly
what "this ablation changed nothing" means.

### The reason the seed count is 16 and not 8

The study was run at 8 seeds first. It gave:

| ablation | 8 seeds | 16 seeds |
|---|---|---|
| no GAE | 1.03x | **1.11x** |
| no advantage normalisation | 1.11x | **1.04x** |

**The ranking of the two small ablations is exactly reversed.** Both runs were
honest, both used the same code, and eight seeds — already above the ">=5"
requirement — was enough to get the order backwards. The conclusion that
survived was only the large one: clipping dominates. Small effects need either
more seeds or an explicit statement that they were not resolved, and this repo
gives the p-values so that the reader can see which is which.

---

## LQR vs PPO where it counts

`compare.py`. Both controllers score 500.0/500 on the nominal task. Step 2
already showed return saturates and cannot rank anything competent, so the
comparison is run on the basin instead — the same bisection and the same
widened-initial-condition sweep, same plant, same termination test, LQR gain
against all 16 trained baseline checkpoints.

| | LQR | PPO, median over 16 seeds | PPO best seed |
|---|---|---|---|
| nominal return, 100 episodes | 500.0 | 500.0 | 500.0 |
| critical `theta_dot0` | **2.169 rad/s** | 1.082 rad/s | 2.004 rad/s |
| critical `x_dot0` | **2.435 m/s** | 0.824 m/s | 1.935 m/s |

Failures out of 200 episodes with the initial-velocity box widened (positions
stay at the standard ±0.05):

| initial velocities | LQR | PPO median | PPO worst seed |
|---|---|---|---|
| U(-0.5, 0.5) | **0** | 1 | 22 |
| U(-1.0, 1.0) | **0** | 48 | 83 |
| U(-1.5, 1.5) | **6** | 88 | 115 |
| U(-2.0, 2.0) | **28** | 116 | 138 |
| U(-2.5, 2.5) | **61** | 140 | 158 |

**PPO's basin is roughly half the LQR's on angular rate and a third on cart
velocity, at identical nominal return.** It learned the initial-state
distribution it was trained on — `U(-0.05, 0.05)` on every state — and nothing
outside it, because nothing outside it was ever sampled and the reward never
asked. Spread across seeds is large: the best seed nearly matches LQR on
`theta_dot0`, the worst is a quarter of it, and the nominal metric calls all
sixteen identical.

This is the result worth keeping from the whole project. A policy that holds the
nominal task with a small basin is a policy that falls over the first time a real
plant hands it a disturbance the training distribution did not contain. Fixing
that is what domain randomisation is for, and the measurement above is the thing
domain randomisation would have to move.

---

## Watching it

`cartpole.xml` + `visualize.py`. MuJoCo is used as a **renderer only** —
`mj_forward` for kinematics, never `mj_step`. `visualize.py` integrates the same
numpy plant from `cartpole.py` and writes the resulting `(x, theta)` into
`qpos`. Letting MuJoCo integrate would silently be a different plant, and every
number in this repo is measured against the numpy one.

```bash
source ~/personal/ml/env.sh
python visualize.py --controller lqr
python visualize.py --controller ppo --ckpt runs/study/baseline_seed0.pt
python visualize.py --controller random                      # dies in ~20 steps
python visualize.py --controller lqr --record out/frames     # offscreen, no window
```

### Reproducing the basin result in three windows

Same disturbance, applied from the exact origin, no random init:

```bash
python visualize.py --controller lqr --zero-init --theta-dot0 1.5
python visualize.py --controller ppo --ckpt runs/study/baseline_seed2.pt --zero-init --theta-dot0 1.5
python visualize.py --controller ppo --ckpt runs/study/baseline_seed0.pt --zero-init --theta-dot0 1.5
```

| controller | outcome at `theta_dot0 = 1.5 rad/s` | peak \|theta\| |
|---|---|---|
| LQR | survives 500 steps | 46% of the limit |
| PPO seed 2 | **falls at step 18** (0.36 s) | 113% |
| PPO seed 0 | survives 500 steps | 52% |

Both findings in one experiment: the learned policy is worse than the
model-based one at the median, *and* the spread across seeds is enormous —
critical `theta_dot0` is 2.004 rad/s for seed 0 and 0.831 rad/s for seed 2, from
identical hyperparameters and an identical step budget. All three score 500.0 on
the nominal task and are indistinguishable by return.

`--zero-init` matters. Without it the random ±0.05 init is still present, and a
negative `theta0` paired with a positive `theta_dot0` is the pole rotating *back
towards* upright — easier, not harder. Every basin number in this README is
bisected from the origin, so reproducing them needs the flag. `baseline_seed0`
is also the best of the 16 seeds, not a typical one; it is the default
checkpoint only because it is the first.

The MJCF matches the physical parameters (cart 1.0 kg, pole 0.1 kg, half-length
0.5 m, rail ±2.4 m, red markers at the termination limits) so that a step-6
continuous-action version could hand the dynamics to MuJoCo — but that would be
a different environment and would need its own baseline. GL is software
(llvmpipe) on this box, so the window frame rate is modest; the offscreen path
uses matplotlib to write PNGs because imageio is not in the shared venv and this
repo installs nothing.

---

## What steps 3–5 do not prove

- Nothing here transfers to a task with a real exploration problem. Cart-pole
  rewards every timestep and a random policy already gets 22; the ablation
  ordering would very likely change on a sparse-reward task.
- The ablations are one-at-a-time. Interactions were not measured.
- One hyperparameter configuration was searched over 8 candidates on 4 seeds.
  A different configuration could change the ablation magnitudes, though the
  clipping result is large enough that it is unlikely to flip.
- `no_clip` failing is measured at *this* learning rate and epoch count. Clipping
  matters because of the 10 epochs; at 1 epoch it would matter much less.
- The basin comparison uses the greedy (argmax) policy. The stochastic policy
  used during training has a different, smaller basin.
- Step 6 repeats the ablations on a different *action space*, not a different
  task. The plant, the reward and the horizon are the same, so it tests whether
  the conclusion is an artefact of the categorical head — not whether it holds
  anywhere else.

---

## Step 6 (stretch) — continuous actions on MuJoCo

`cartpole_mj.xml`, `mj_cartpole.py`, `ppo_continuous.py`. 6/6 environment
checks pass, 16 seeds and 3 ablations measured.

Two things change from steps 1–5 at once, so both are pinned down separately.

### The plant is identical — verified to 1.4e-14

`cartpole_mj.xml` sets the inertias by hand (`inertiafromgeom="false"`) instead
of letting MuJoCo derive them from a capsule geom. A capsule adds hemispherical
caps and a cylinder adds a `3r²/12` term; the `4/3` in `cartpole.py` is exactly
the uniform-thin-rod assumption, so deriving inertia from the geom would have
made these two plants *nearly* the same object, which is the worst possible
state to be in.

| check | result |
|---|---|
| V1. MuJoCo `qacc` == `cartpole.accelerations()`, 5000 random states | max abs **1.42e-14**, max rel **2.23e-12** |

The two simulators are describing the same rigid body. Anything that differs
from here on is the integrator or the action space, and cannot be blamed on the
model.

### The integrator is different, and that is not reconcilable

MuJoCo's `Euler` is semi-implicit: velocity first, then position advanced with
the **new** velocity. `cartpole.py` uses the explicit ordering because that is
what CartPole-v1 does. So:

| | |
|---|---|
| V2. one-step gap, same state and same force | median \|dθ\| **5.84e-3 rad**, max **7.02e-3** (3.35% of the 12° limit) |
| | median \|dx\| 3.90e-3 m |

| V3. energy drift, zero control, θ₀ = 0.5 rad | at 50 steps | at 500 steps |
|---|---|---|
| MuJoCo semi-implicit | +6.69% | **+9.85%** |
| our explicit Euler (step 1) | **+0.43%** | +145.0% |

Semi-implicit Euler is worse in the short run and dramatically better in the
long run: its energy error oscillates with bounded amplitude, while the explicit
version's grows. Neither is "the correct plant" — they are two discretisations
of the same continuous system, and this is the reason step 6 gets its own
baseline instead of borrowing step 4's numbers.

### The step-2 LQR gain transfers unchanged

This is the interesting one. The gain was computed against the numpy plant, in
step 2, before this file existed. Applied to the MuJoCo plant without
re-deriving anything:

| V5. step-2 gain `K` on the MuJoCo plant | mean return, 100 episodes |
|---|---|
| bang-bang, sign only, ±10 N | **500.0** (min 500) |
| continuous, clipped to ±10 N | **500.0** (min 500) |

A model-based design survived being moved to a different integrator, at zero
cost, because it depends on the plant and not on the discretisation. That is the
smallest honest version of a sim-to-real argument that this repo can make, and
it is the one thing the learned policy in step 4 was never asked to do.

Other contract checks: V4a truncation at exactly the cap with reward 1.0 each
step; V4b `step()` after the end raises; V4c same seed bitwise identical;
V4d a uniform-random continuous policy scores **26.59 ± 0.28** over 2000
episodes against **22.09 ± 0.12** for the discrete env — a random continuous
action averages less force than a random ±10 N bang, so it perturbs the pole
less and survives slightly longer.

### The Gaussian policy

`ppo_continuous.py` imports `compute_gae` and `layer_init` from `ppo.py` rather
than copying them, so the two implementations cannot drift apart. Only the head
changes: `Categorical(logits)` becomes `Normal(mu(s), exp(log_std))` with
`log_std` a free parameter, log-prob summed over action dimensions, and Gaussian
differential entropy.

`ent_coef` defaults to **0.0** here and 0.01 in `ppo.py`. Gaussian differential
entropy is unbounded below and can go negative, so an entropy bonus pushes sigma
up with no floor — the same coefficient does not mean the same thing in the two
files.

**Action clipping is biased and the size of the bias is reported.** The Gaussian
has support on all of R; the actuator saturates at ±1. The action is clipped at
the environment boundary while the log-probability is computed on the unclipped
sample, so every out-of-range sample is credited with a density it did not act
under. The honest alternatives are a tanh-squashed policy with the
change-of-variables correction, or a Beta policy on the bounded interval;
neither is used. Instead every run reports its saturation fraction.

Search on seeds 100–101 (150k steps, disjoint from any study seed):

| init `log_std` | lr | solved | steps-to-threshold | final sigma | saturation |
|---|---|---|---|---|---|
| **−1.0** | **2e-3** | **2/2** | **63,392 / 65,528** | **0.133** | **0.5%** |
| −0.5 | 2e-3 | 2/2 | 62,632 / 72,504 | 0.238 | 6.9% |
| −1.5 | 1e-3 | 2/2 | 75,736 / 76,456 | 0.174 | 0.0% |
| −1.0 | 1e-3 | 2/2 | 77,664 / 80,376 | 0.339 | 5.8% |
| −0.5 | 1e-3 | 2/2 | 80,216 / 69,264 | 0.435 | 12.3% |
| 0.0 | 1e-3 | 2/2 | 112,936 / 78,800 | 0.785 | **36.7%** |
| −1.0 | 5e-4 | 0/2 | — | 0.432 | 3.1% |

The `init_log_std = 0` row is the point of the table: sigma = 1 on a ±1 action
range means **37% of sampled actions are outside the actuator range**, the
policy spends most of its probability mass on actions the plant cannot execute,
and it costs roughly 50% more environment steps. Initialising the standard
deviation is not a detail on a bounded action space.

### Seed study — 16 seeds, Gaussian policy on MuJoCo

Same protocol as step 4: seeds 0-15, 150k-step budget, "solved" = mean return
>= 475 over 100 consecutive episodes with the episode capped at 500 steps.

| | solved | steps-to-threshold median | IQR | greedy eval (100 ep) |
|---|---|---|---|---|
| discrete, `ppo.py` (step 4) | 16/16 | 62,144 | [60,442, 63,448] | 500.0 |
| **continuous, `ppo_continuous.py`** | **16/16** | **63,748** | **[62,008, 65,636]** | **500.0** |

**Moving from two discrete actions to a one-dimensional Gaussian costs 2.6% of
the sample budget on this task, and the IQRs overlap.** That is a smaller gap
than expected. It is not evidence that continuous control is free in general —
it is evidence that on a plant this small, with the action range matched to the
force the discrete policy was already applying, the extra difficulty of learning
a mean *and* a standard deviation is nearly paid for by the finer control
authority. The saturation numbers below are why: the tuned policy barely uses
the continuous range.

16 runs, 73.5 s wall at 4-way parallel, median 21.9 s per run.

### Ablations — the step-5 ranking transfers, and clipping gets worse

Same three switches, same 16 seeds, same budget.

| ablation | solved | steps median | vs baseline | permutation p | final sigma | saturation |
|---|---|---|---|---|---|---|
| baseline | 16/16 | 63,748 | 1.00x | — | 0.136 | 0.5% |
| no advantage normalisation | 16/16 | 64,664 | 1.01x | **0.73** | 0.144 | 0.1% |
| no GAE (lambda = 1) | 16/16 | 71,760 | 1.13x | 0.0032 | 0.218 | 0.8% |
| **no ratio clipping** | **0/16** | **>150,000** | **2.35x** | 0.0008 | 0.088 | **28.2%** |

Side by side with step 5:

| ablation | discrete | continuous |
|---|---|---|
| no advantage normalisation | 1.04x, 16/16, p=0.054 | 1.01x, 16/16, p=0.73 |
| no GAE | 1.11x, 16/16, p=0.0006 | 1.13x, 16/16, p=0.0032 |
| **no ratio clipping** | **2.34x, 8/16 solved** | **2.35x, 0/16 solved** |

**The ordering survives the change of action space; the magnitude of the top
effect does not.** Without clipping the discrete policy still gets half its
seeds to threshold. The Gaussian policy gets none — 0/16, median greedy return
126 against a 500 cap.

The failure mechanism is visible in the two right-hand columns and it is
specific to the continuous case. `no_clip` ends with the *smallest* sigma of any
config (0.088) and by far the *largest* saturation fraction (28.2%, up to 80.6%
on the worst seed). A Gaussian with sigma = 0.088 only puts 28% of its mass
outside [-1, 1] if the mean sits at roughly +-0.95 — the mean has been driven
onto the actuator rail while the standard deviation collapsed around it. That is
the unclipped update doing exactly what clipping exists to prevent: with an
importance ratio unbounded above on a continuous density, one favourable
minibatch moves the mean as far as the gradient points, and the nine remaining
epochs on that same batch reinforce a policy that is already off the plant's
control range. The discrete policy cannot do this — a categorical ratio is
bounded by `1 / pi_old(a)` on a two-element support, and the worst it can do is
become deterministic between two actions that are both executable.

So the honest step-6 conclusion is narrower than "clipping matters more in
continuous control". It is: **on an action space where the policy can place mass
outside the actuator range, removing the clip is not a slowdown, it is a
failure mode.** One task, one action dimension, one plant.

### Cost

| | wall | runs |
|---|---|---|
| seed study | 73.5 s | 16 |
| seed study + 3 ablations | 426.3 s | 64 |

Both on a quiet box (1-min load 0.32 and 1.88 at entry, block 2 finished at
14:06), 4-way parallel at `nice 10`. Longest single run 26 s, well inside the
10-minute rule.

---

## Box etiquette, and the time it failed

This machine has 8 threads shared with `mujoco-clutter-detect`, which has
priority. Every entry point that starts training calls
`boxcheck.require_quiet_box()` and refuses above a 1-minute load average of 4.0.

That guard originally lived only in `study.py`, and it did not help. The
step-6 hyperparameter sweep was launched from a bare shell loop, which never
called `study.py`, and it ran 11:30–11:33 straight through a `train_det.py`
that had started at 11:25 and was using 770% CPU. A guard on the front door is
not a guard, so it now lives in `boxcheck.py` and is called by `ppo.py`,
`ppo_continuous.py` and `study.py` alike, and it prints the offending process.

Consequences, stated rather than buried:

- **The step-6 search wall-clock numbers are contaminated** and are not
  reported above. The *learning* outcomes are not: runs are seeded and
  single-threaded, and two identical configurations submitted under different
  tags during that window returned bitwise-identical steps-to-threshold
  (80,216 / 69,264), which is the evidence that contention moved the clock and
  nothing else.
- **`mj_cartpole.py`'s V6 throughput figure was measured under load, and
  re-measuring proved it.** Contended: 73,962 env steps/s. Quiet box, same
  binary, same check: **145,458 steps/s** — 1.97x, against a numpy env at
  373,919/s, so MuJoCo is 2.6x slower rather than the 5x first recorded. A
  throughput number carries the load average it was taken under or it carries
  nothing.
- **Steps 1–5 are clean.** The 64-run study finished at 11:19, six minutes
  before the detection job started, and `study.py`'s guard passed at launch.

## Running

```bash
source ~/personal/ml/env.sh
python verify_env.py          # step 1, ~6 s, one core
python lqr.py                 # step 2, ~30 s, one core
python ppo.py --seed 0        # step 3, one 150k-step run, ~11 s
python study.py --mode all    # steps 3-5, 96 runs, ~8 min, 4-way parallel
python compare.py             # LQR vs PPO basin comparison, ~3 min
python visualize.py --controller lqr        # MuJoCo viewer
python mj_cartpole.py         # step 6 env, 6/6 checks, ~2 min
python ppo_continuous.py --seed 0                            # one Gaussian run
python study.py --family continuous --mode ablations --out runs_c   # step 6, 64 runs, ~7 min
```

No dependencies beyond numpy, already in the shared `~/personal/ml/.venv`.
Nothing in this repo installs anything: the venv is shared with the detection
project and an install here changes that project's environment mid-run.
