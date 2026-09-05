"""Cart-pole dynamics, hand-written. No gymnasium, no scipy. numpy only.

Why this file exists at all
---------------------------
PPO will fail to learn several times before step 5 of this repo is finished.
Every time it does, the first question is "is it the algorithm or is it the
environment". That question is only cheap to answer if the environment has
already been verified independently. So this is the reference implementation:
scalar, slow, readable, and checked against the published equations by
`verify_env.py` before a single line of RL was written.

The dynamics are Barto, Sutton & Anderson (1983), "Neuronlike adaptive elements
that can solve difficult learning control problems", IEEE SMC-13(5), appendix.
Theta is measured from upright, positive x is to the right, positive theta leans
the pole toward +x.

    temp = (F + m*l*thetadot^2*sin(theta)) / (M + m)
    thetaddot = (g*sin(theta) - cos(theta)*temp)
                / (l * (4/3 - m*cos(theta)^2/(M + m)))
    xddot = temp - m*l*thetaddot*cos(theta) / (M + m)

`l` is the HALF-length of the pole. The 4/3 is not a fudge factor: it is
1 + I_cm/(m*l^2) for a uniform rod, whose moment of inertia about its own centre
is m*(2l)^2/12 = m*l^2/3. Getting `l` wrong by a factor of two is the single most
common bug in a hand-written cart-pole, and check 3 in the verifier exists
specifically to catch it.

Bit-compatibility with CartPole-v1
----------------------------------
The integrator is EXPLICIT Euler -- position is advanced with the OLD velocity --
because that is what gymnasium's CartPole-v1 does by default. Semi-implicit Euler
is the better integrator and it is deliberately not used here: the whole point of
this repo is a number ("solved = mean return >= 475 over 100 episodes, 500-step
cap") that is comparable to everyone else's published number, and that number is
defined under v1's integrator. Explicit Euler injects energy; `verify_env.py`
measures how much and the README reports it rather than quietly fixing it.

Deviations from CartPole-v1, complete list:
  1. Observations are float64, not float32.
  2. Stepping a finished episode raises RuntimeError instead of warning and
     returning reward 0. Silently rewarding a dead episode is a PPO rollout bug
     that is very hard to see in a learning curve.
  3. `terminated` and `truncated` are returned separately and must stay separate.
     Collapsing them into one `done` flag makes the value target at the 500-step
     cap treat a perfectly healthy state as worth zero future return. That single
     mistake is worth ~100 points of final performance and is the reason this
     env refuses to expose a combined flag at all.
"""
from dataclasses import dataclass
import math

import numpy as np

# --- episode contract, CartPole-v1 convention ---
X_THRESHOLD = 2.4                          # m, cart position at which the episode fails
THETA_THRESHOLD = 12 * 2 * math.pi / 360   # rad, 12 degrees
MAX_EPISODE_STEPS = 500                    # truncation, NOT termination
INIT_RANGE = 0.05                          # all four state components ~ U(-0.05, 0.05)

# state vector layout, in this order everywhere in the repo
X, X_DOT, THETA, THETA_DOT = 0, 1, 2, 3


@dataclass(frozen=True)
class CartPoleParams:
    """Physical constants. Frozen so a run cannot silently mutate them.

    Exposed as a parameter object rather than module constants because the
    verifier needs to push masscart to 1e9 (fixed-cart limit) and step 2 needs
    them for the linearisation.
    """
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5      # HALF the pole length
    force_mag: float = 10.0
    tau: float = 0.02        # s, control period

    @property
    def total_mass(self) -> float:
        return self.masscart + self.masspole

    @property
    def polemass_length(self) -> float:
        return self.masspole * self.length


DEFAULT = CartPoleParams()


def accelerations(state, force, p=DEFAULT):
    """(xddot, thetaddot) from the published scalar equations.

    This is the hand-eliminated form. `verify_env.py` re-derives the same two
    numbers by assembling the Lagrangian mass matrix and calling np.linalg.solve,
    which shares no algebra with this function.
    """
    theta, theta_dot = state[THETA], state[THETA_DOT]
    costheta, sintheta = math.cos(theta), math.sin(theta)

    temp = (force + p.polemass_length * theta_dot ** 2 * sintheta) / p.total_mass
    thetaacc = (p.gravity * sintheta - costheta * temp) / (
        p.length * (4.0 / 3.0 - p.masspole * costheta ** 2 / p.total_mass)
    )
    xacc = temp - p.polemass_length * thetaacc * costheta / p.total_mass
    return xacc, thetaacc


def deriv(state, force, p=DEFAULT):
    """Continuous-time state derivative. Used by the verifier's RK4, not by the env."""
    xacc, thetaacc = accelerations(state, force, p)
    return np.array([state[X_DOT], xacc, state[THETA_DOT], thetaacc], dtype=np.float64)


def euler_step(state, force, p=DEFAULT):
    """One EXPLICIT Euler step, exactly as CartPole-v1 does it.

    Note the ordering: x is advanced using the OLD x_dot, then x_dot is advanced.
    Swapping those two lines gives semi-implicit Euler, a different (better)
    integrator and a different environment.
    """
    x, x_dot, theta, theta_dot = state
    xacc, thetaacc = accelerations(state, force, p)
    return np.array(
        [
            x + p.tau * x_dot,
            x_dot + p.tau * xacc,
            theta + p.tau * theta_dot,
            theta_dot + p.tau * thetaacc,
        ],
        dtype=np.float64,
    )


def energy(state, p=DEFAULT):
    """Total mechanical energy, written from the Lagrangian, not from `accelerations`.

    KE = 1/2 (M+m) xdot^2 + m l xdot thetadot cos(theta) + 2/3 m l^2 thetadot^2
    PE = m g l cos(theta)

    The 2/3 is 1/2 * (4/3): the pole's translational and rotational kinetic
    energy combined. Conserved exactly when F = 0.
    """
    _, x_dot, theta, theta_dot = state
    m, l = p.masspole, p.length
    ke = (
        0.5 * p.total_mass * x_dot ** 2
        + m * l * x_dot * theta_dot * math.cos(theta)
        + (2.0 / 3.0) * m * l ** 2 * theta_dot ** 2
    )
    pe = m * p.gravity * l * math.cos(theta)
    return ke + pe


def momentum(state, p=DEFAULT):
    """Horizontal momentum d(L)/d(xdot). Conserved when F = 0, because x is cyclic."""
    _, x_dot, theta, theta_dot = state
    return p.total_mass * x_dot + p.polemass_length * theta_dot * math.cos(theta)


class CartPoleEnv:
    """CartPole-v1 semantics, hand-written.

    Owns its own Generator. Nothing in this repo may touch the global numpy RNG:
    the seed study in step 4 is worthless if two "different seeds" share state.
    """

    n_actions = 2
    obs_dim = 4

    def __init__(self, params=DEFAULT, max_episode_steps=MAX_EPISODE_STEPS, seed=None):
        self.p = params
        self.max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng(seed)
        self.state = None
        self._steps = 0
        self._done = True

    def reset(self, seed=None):
        """Reseed only if `seed` is given; otherwise continue the existing stream.

        Calling reset(seed=k) once and then reset() per episode is the correct
        pattern for a seeded run: episodes differ, runs are reproducible.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self._rng.uniform(-INIT_RANGE, INIT_RANGE, size=4).astype(np.float64)
        self._steps = 0
        self._done = False
        return self.state.copy(), {}

    def step(self, action):
        if self._done:
            raise RuntimeError("step() after the episode ended. Call reset() first.")
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action!r}")

        force = self.p.force_mag if action == 1 else -self.p.force_mag
        self.state = euler_step(self.state, force, self.p)
        self._steps += 1

        x, theta = self.state[X], self.state[THETA]
        terminated = bool(abs(x) > X_THRESHOLD or abs(theta) > THETA_THRESHOLD)
        truncated = bool(not terminated and self._steps >= self.max_episode_steps)
        self._done = terminated or truncated

        # Reward is 1.0 on every step INCLUDING the one that terminates. That is
        # the v1 convention and it is what makes the max return equal the step cap.
        return self.state.copy(), 1.0, terminated, truncated, {}
