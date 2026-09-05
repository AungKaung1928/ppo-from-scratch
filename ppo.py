"""
Step 3 -- PPO from scratch. Single file, categorical policy, no RL library.

Nothing here is imported from stable-baselines3, rllib or cleanrl. The point of
the project is the internals, so the four things that are actually PPO --
GAE, advantage normalisation, the clipped surrogate, and the multi-epoch
minibatch update -- are written out longhand and each one can be switched off
from the command line, which is what step 5 measures.

THE FOUR PIECES, AND WHY EACH IS THERE
--------------------------------------
1. GAE(gamma, lam). The advantage estimator interpolates between the one-step
   TD residual (lam=0: low variance, biased by whatever the critic gets wrong)
   and the Monte-Carlo return minus baseline (lam=1: unbiased, variance grows
   with horizon). Computed by the backward recursion

       A_t = delta_t + gamma*lam*(1 - done_t)*A_{t+1},
       delta_t = r_t + gamma*(1 - done_t)*V(s_{t+1}) - V(s_t)

   `--no-gae` sets lam = 1.0, which is exactly the plain Monte-Carlo advantage.
   That is the honest form of the ablation: GAE(gamma, 1) IS the no-GAE case,
   so the switch changes one number and nothing else.

2. Advantage normalisation. Per minibatch, (A - mean)/(std + 1e-8). It makes
   the effective step size independent of the reward scale -- on cart-pole the
   advantage magnitude grows as the policy improves and episodes get longer, so
   without it the same learning rate means something different at step 10k and
   at step 100k. `--no-advnorm` removes it.

3. The clipped surrogate. maximise min(rho_t A_t, clip(rho_t, 1-e, 1+e) A_t)
   with rho_t = pi_new/pi_old. This is the whole reason PPO can reuse a batch
   for several epochs: the clip removes the incentive to keep pushing rho past
   the trust region, so gradients vanish for samples that have already moved
   far enough. `--no-clip` replaces it with the raw importance-weighted
   objective rho_t A_t, which is a correct policy gradient for ONE epoch and
   an increasingly wrong one for four.

4. Multi-epoch minibatch updates. Without them PPO is just A2C with extra
   arithmetic. They are what makes the clip load-bearing.

TRUNCATION
----------
Step 1 kept `terminated` and `truncated` separate, and this is the file that
needed it. At the 500-step cap the pole has NOT fallen -- the episode was cut
off by the clock. Bootstrapping V(s_final) there is correct; treating it as a
terminal state teaches the critic that surviving 500 steps is worth 0 future
return, which is the single most common silent bug in a hand-written PPO.

Implemented by folding the bootstrap into the reward of the truncated step:

    r_t += gamma * V(s_final)      when truncated and not terminated

and then treating the step as an episode boundary for GAE, which it is,
because the next observation comes from a fresh reset.

HYPERPARAMETERS
---------------
lr = 2e-3 and 10 epochs are not defaults copied from anywhere -- they came from
a small search run on seeds 100-103, which are deliberately disjoint from the
seeds 0-7 used for the study in step 4. The first configuration I wrote
(lr 3e-4, 4 epochs) never reached threshold inside 150k steps and reported
clipfrac 0.000 on every update: the policy was moving so little per update that
the clip never engaged. That is the search, and the table is in the README. Any
number here that was tuned on the seeds it is then reported on is worthless,
which is why the split exists.

THREADS
-------
torch.set_num_threads(1) on purpose. The networks are 64x64; intra-op threading
costs more in synchronisation than it saves, and this box shares 8 threads with
another training job. Measured both ways -- the number is in the README.
"""

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from boxcheck import require_quiet_box
from cartpole import CartPoleEnv

torch.set_num_threads(1)


@dataclass
class Config:
    seed: int = 0
    total_timesteps: int = 150_000
    num_envs: int = 8
    num_steps: int = 128
    lr: float = 2e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    # ablations
    use_gae: bool = True
    use_advnorm: bool = True
    use_clip: bool = True
    # bookkeeping
    solve_return: float = 475.0
    solve_window: int = 100
    tag: str = "baseline"
    out: str = "runs"

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches


def layer_init(layer, std=math.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Separate trunks. Sharing one saves parameters we do not need to save and
    couples the value loss scale to the policy gradient, which is one more knob
    to get wrong in an ablation study."""

    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0))
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(x)


def compute_gae(rewards, values, dones, last_value, last_done, gamma, lam):
    """Backward recursion. `dones` marks an episode boundary AT that step, so
    the bootstrap through it is cut. Truncation bootstraps are already folded
    into `rewards` by the caller."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(N)
    next_value, next_nonterminal = last_value, 1.0 - last_done
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        running = delta + gamma * lam * next_nonterminal * running
        adv[t] = running
        next_value, next_nonterminal = values[t], 1.0 - dones[t]
    return adv, adv + values


def greedy_eval(agent, episodes=100, seed=12345):
    env = CartPoleEnv(seed=seed)
    out = np.empty(episodes)
    with torch.no_grad():
        for i in range(episodes):
            obs, _ = env.reset()
            total = 0.0
            while True:
                logits = agent.actor(torch.as_tensor(obs, dtype=torch.float32))
                obs, r, term, trunc, _ = env.step(int(torch.argmax(logits)))
                total += r
                if term or trunc:
                    break
            out[i] = total
    return out


def train(cfg: Config, verbose=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)          # only touches nothing the envs use; see step 1 check 5b
    envs = [CartPoleEnv(seed=cfg.seed * 1000 + i) for i in range(cfg.num_envs)]
    agent = ActorCritic(CartPoleEnv.obs_dim, CartPoleEnv.n_actions, cfg.hidden)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    obs = torch.stack([torch.as_tensor(e.reset()[0], dtype=torch.float32) for e in envs])
    done = torch.zeros(cfg.num_envs)
    ep_return = np.zeros(cfg.num_envs)

    window = deque(maxlen=cfg.solve_window)
    steps_to_threshold = None
    trace = []
    global_step = 0
    n_updates = cfg.total_timesteps // cfg.batch_size
    t_start = time.perf_counter()

    b_obs = torch.zeros(cfg.num_steps, cfg.num_envs, CartPoleEnv.obs_dim)
    b_act = torch.zeros(cfg.num_steps, cfg.num_envs, dtype=torch.long)
    b_logp = torch.zeros(cfg.num_steps, cfg.num_envs)
    b_rew = torch.zeros(cfg.num_steps, cfg.num_envs)
    b_done = torch.zeros(cfg.num_steps, cfg.num_envs)
    b_val = torch.zeros(cfg.num_steps, cfg.num_envs)

    for update in range(1, n_updates + 1):
        if cfg.anneal_lr:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - (update - 1.0) / n_updates)

        for t in range(cfg.num_steps):
            global_step += cfg.num_envs
            b_obs[t], b_done[t] = obs, done
            with torch.no_grad():
                action, logp, _, value = agent.act(obs)
            b_act[t], b_logp[t], b_val[t] = action, logp, value

            next_obs = np.empty((cfg.num_envs, CartPoleEnv.obs_dim), dtype=np.float32)
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(int(action[i]))
                ep_return[i] += r
                if trunc and not term:
                    # cut off by the clock, not by falling over: bootstrap.
                    with torch.no_grad():
                        r += cfg.gamma * float(
                            agent.value(torch.as_tensor(o, dtype=torch.float32)))
                b_rew[t, i] = r
                if term or trunc:
                    window.append(ep_return[i])
                    if (steps_to_threshold is None and len(window) == cfg.solve_window
                            and float(np.mean(window)) >= cfg.solve_return):
                        steps_to_threshold = global_step
                    ep_return[i] = 0.0
                    o, _ = e.reset()
                    done[i] = 1.0
                else:
                    done[i] = 0.0
                next_obs[i] = o
            obs = torch.as_tensor(next_obs)

        with torch.no_grad():
            last_value = agent.value(obs)
        lam = cfg.gae_lambda if cfg.use_gae else 1.0
        adv, ret = compute_gae(b_rew, b_val, b_done, last_value, done, cfg.gamma, lam)

        f_obs = b_obs.reshape(-1, CartPoleEnv.obs_dim)
        f_act, f_logp = b_act.reshape(-1), b_logp.reshape(-1)
        f_adv, f_ret = adv.reshape(-1), ret.reshape(-1)
        idx = np.arange(cfg.batch_size)
        clipfracs = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[s:s + cfg.minibatch_size]
                _, newlogp, entropy, newval = agent.act(f_obs[mb], f_act[mb])
                logratio = newlogp - f_logp[mb]
                ratio = logratio.exp()
                mb_adv = f_adv[mb]
                if cfg.use_advnorm:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                if cfg.use_clip:
                    unclipped = ratio * mb_adv
                    clipped = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                    pg_loss = -torch.min(unclipped, clipped).mean()
                else:
                    pg_loss = -(ratio * mb_adv).mean()
                v_loss = 0.5 * ((newval - f_ret[mb]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

        rolling = float(np.mean(window)) if window else 0.0
        trace.append({"step": global_step, "rolling100": rolling,
                      "clipfrac": float(np.mean(clipfracs))})
        if verbose and update % 10 == 0:
            print(f"  upd {update:>4}/{n_updates}  step {global_step:>7}  "
                  f"rolling100 {rolling:7.1f}  clipfrac {np.mean(clipfracs):.3f}")

    wall = time.perf_counter() - t_start
    greedy = greedy_eval(agent)
    result = {
        "config": asdict(cfg),
        "steps_to_threshold": steps_to_threshold,
        "solved": steps_to_threshold is not None,
        "final_rolling100": float(np.mean(window)) if window else 0.0,
        "greedy_eval_mean": float(greedy.mean()),
        "greedy_eval_min": float(greedy.min()),
        "wall_clock_s": wall,
        "steps_per_s": cfg.total_timesteps / wall,
        "trace": trace,
    }
    os.makedirs(cfg.out, exist_ok=True)
    stem = os.path.join(cfg.out, f"{cfg.tag}_seed{cfg.seed}")
    with open(stem + ".json", "w") as f:
        json.dump(result, f)
    torch.save({"state_dict": agent.state_dict(), "config": asdict(cfg)},
               stem + ".pt")
    return result


def load_agent(path):
    ckpt = torch.load(path, weights_only=False)
    c = ckpt["config"]
    agent = ActorCritic(CartPoleEnv.obs_dim, CartPoleEnv.n_actions, c["hidden"])
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent, c


def parse():
    p = argparse.ArgumentParser()
    d = Config()
    for k, v in asdict(d).items():
        if isinstance(v, bool):
            continue
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--no-gae", action="store_true")
    p.add_argument("--no-advnorm", action="store_true")
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="start even if the box is busy. Do not.")
    a = p.parse_args()
    require_quiet_box(a.force, quiet=a.quiet)
    cfg = Config(**{k: getattr(a, k) for k in asdict(d) if hasattr(a, k)})
    cfg.use_gae = not a.no_gae
    cfg.use_advnorm = not a.no_advnorm
    cfg.use_clip = not a.no_clip
    cfg.anneal_lr = not a.no_anneal_lr
    return cfg, a.quiet


if __name__ == "__main__":
    cfg, quiet = parse()
    print(f"=== ppo  tag={cfg.tag}  seed={cfg.seed}  "
          f"gae={cfg.use_gae} advnorm={cfg.use_advnorm} clip={cfg.use_clip} ===")
    r = train(cfg, verbose=not quiet)
    print(f"  steps_to_threshold : {r['steps_to_threshold']}")
    print(f"  final rolling100   : {r['final_rolling100']:.1f}")
    print(f"  greedy eval (100)  : {r['greedy_eval_mean']:.1f} (min {r['greedy_eval_min']:.0f})")
    print(f"  wall clock         : {r['wall_clock_s']:.1f} s "
          f"({r['steps_per_s']:.0f} env steps/s)")
