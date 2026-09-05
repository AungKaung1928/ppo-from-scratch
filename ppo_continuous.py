"""
Step 6 (stretch) -- PPO with a Gaussian policy on the MuJoCo cart-pole.

WHAT CHANGES FROM ppo.py, AND WHAT DOES NOT
-------------------------------------------
Unchanged, and imported rather than copied so that they cannot drift: the GAE
recursion, the orthogonal init, the clipped surrogate, the truncation bootstrap,
the multi-epoch minibatch update, the "solved" definition and the
steps-to-threshold bookkeeping.

Changed, and only this:

  the head          Categorical(logits) becomes Normal(mu(s), exp(log_std)).
                    log_std is a free parameter, not a function of the state.
                    A state-dependent std is the usual next step and is NOT
                    taken here: with one action dimension and a plant this
                    small it adds parameters that the ablation study would then
                    have to control for.

  the log-prob      summed over action dimensions, because the Gaussian is
                    diagonal and the joint log-density is the sum. With
                    act_dim = 1 the sum is a no-op, but writing it as a sum is
                    the difference between code that generalises and code that
                    silently breaks the first time act_dim > 1.

  the entropy       Gaussian differential entropy, 0.5*log(2*pi*e*sigma^2) per
                    dimension. Unlike the categorical case this is unbounded
                    below and can go negative, so an entropy bonus here pushes
                    sigma up without a floor. That is why ent_coef defaults to
                    0.0 in this file and 0.01 in ppo.py -- the same coefficient
                    does not mean the same thing.

ACTION CLIPPING, STATED RATHER THAN HIDDEN
------------------------------------------
The Gaussian has support on all of R; the actuator saturates at +-1. The action
is clipped at the environment boundary while the log-probability is computed on
the UNCLIPPED sample. That is the common choice and it is biased: every sample
outside [-1,1] is credited with a density it did not act under. The honest
alternatives are a squashed (tanh) policy with the change-of-variables
correction, or a Beta policy on the bounded interval. Neither is used here, and
the measured saturation fraction is reported so the size of the sin is visible
instead of assumed small.
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
from mj_cartpole import MjCartPoleEnv
from ppo import compute_gae, layer_init

torch.set_num_threads(1)


@dataclass
class ConfigC:
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
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    init_log_std: float = -1.0
    use_gae: bool = True
    use_advnorm: bool = True
    use_clip: bool = True
    solve_return: float = 475.0
    solve_window: int = 100
    tag: str = "cont"
    out: str = "runs_c"

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches


class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, init_log_std=0.0):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0))
        self.mu = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        mean = self.mu(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        # sum over action dims: diagonal Gaussian, joint log-density is the sum
        return (action, dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1), self.value(x))


def eval_mean_action(agent, episodes=100, seed=54321):
    env = MjCartPoleEnv(seed=seed)
    out = np.empty(episodes)
    with torch.no_grad():
        for i in range(episodes):
            obs, _ = env.reset()
            tot = 0.0
            while True:
                a = agent.mu(torch.as_tensor(obs, dtype=torch.float32))
                obs, r, te, tr, _ = env.step(a.numpy())
                tot += r
                if te or tr:
                    break
            out[i] = tot
    return out


def train(cfg: ConfigC, verbose=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    envs = [MjCartPoleEnv(seed=cfg.seed * 1000 + i) for i in range(cfg.num_envs)]
    A, O = MjCartPoleEnv.act_dim, MjCartPoleEnv.obs_dim
    agent = GaussianActorCritic(O, A, cfg.hidden, cfg.init_log_std)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    obs = torch.stack([torch.as_tensor(e.reset()[0], dtype=torch.float32) for e in envs])
    done = torch.zeros(cfg.num_envs)
    ep_return = np.zeros(cfg.num_envs)
    window = deque(maxlen=cfg.solve_window)
    steps_to_threshold, global_step, trace = None, 0, []
    n_updates = cfg.total_timesteps // cfg.batch_size
    t0 = time.perf_counter()
    saturated = 0
    total_actions = 0

    b_obs = torch.zeros(cfg.num_steps, cfg.num_envs, O)
    b_act = torch.zeros(cfg.num_steps, cfg.num_envs, A)
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
            a_np = action.numpy()
            saturated += int((np.abs(a_np) > 1.0).sum())
            total_actions += a_np.size

            nxt = np.empty((cfg.num_envs, O), dtype=np.float32)
            for i, e in enumerate(envs):
                o, r, te, tr, _ = e.step(a_np[i])
                ep_return[i] += r
                if tr and not te:
                    with torch.no_grad():
                        r += cfg.gamma * float(
                            agent.value(torch.as_tensor(o, dtype=torch.float32)))
                b_rew[t, i] = r
                if te or tr:
                    window.append(ep_return[i])
                    if (steps_to_threshold is None
                            and len(window) == cfg.solve_window
                            and float(np.mean(window)) >= cfg.solve_return):
                        steps_to_threshold = global_step
                    ep_return[i] = 0.0
                    o, _ = e.reset()
                    done[i] = 1.0
                else:
                    done[i] = 0.0
                nxt[i] = o
            obs = torch.as_tensor(nxt)

        with torch.no_grad():
            last_value = agent.value(obs)
        lam = cfg.gae_lambda if cfg.use_gae else 1.0
        adv, ret = compute_gae(b_rew, b_val, b_done, last_value, done, cfg.gamma, lam)

        f_obs, f_act = b_obs.reshape(-1, O), b_act.reshape(-1, A)
        f_logp, f_adv, f_ret = b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        idx = np.arange(cfg.batch_size)
        clipfracs = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[s:s + cfg.minibatch_size]
                _, newlogp, entropy, newval = agent.act(f_obs[mb], f_act[mb])
                ratio = (newlogp - f_logp[mb]).exp()
                mb_adv = f_adv[mb]
                if cfg.use_advnorm:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                if cfg.use_clip:
                    pg_loss = -torch.min(
                        ratio * mb_adv,
                        torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                    ).mean()
                else:
                    pg_loss = -(ratio * mb_adv).mean()
                v_loss = 0.5 * ((newval - f_ret[mb]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())

        rolling = float(np.mean(window)) if window else 0.0
        trace.append({"step": global_step, "rolling100": rolling,
                      "clipfrac": float(np.mean(clipfracs)),
                      "std": float(agent.log_std.detach().exp().mean())})
        if verbose and update % 10 == 0:
            print(f"  upd {update:>4}/{n_updates}  step {global_step:>7}  "
                  f"rolling100 {rolling:7.1f}  clipfrac {np.mean(clipfracs):.3f}  "
                  f"sigma {float(agent.log_std.detach().exp().mean()):.3f}")

    wall = time.perf_counter() - t0
    ev = eval_mean_action(agent)
    result = {
        "config": asdict(cfg),
        "steps_to_threshold": steps_to_threshold,
        "solved": steps_to_threshold is not None,
        "final_rolling100": float(np.mean(window)) if window else 0.0,
        "greedy_eval_mean": float(ev.mean()),
        "greedy_eval_min": float(ev.min()),
        "final_sigma": float(agent.log_std.detach().exp().mean()),
        "saturation_fraction": saturated / max(total_actions, 1),
        "wall_clock_s": wall,
        "trace": trace,
    }
    os.makedirs(cfg.out, exist_ok=True)
    stem = os.path.join(cfg.out, f"{cfg.tag}_seed{cfg.seed}")
    with open(stem + ".json", "w") as f:
        json.dump(result, f)
    torch.save({"state_dict": agent.state_dict(), "config": asdict(cfg)}, stem + ".pt")
    return result


def load_agent(path):
    ckpt = torch.load(path, weights_only=False)
    c = ckpt["config"]
    ag = GaussianActorCritic(MjCartPoleEnv.obs_dim, MjCartPoleEnv.act_dim,
                             c["hidden"], c["init_log_std"])
    ag.load_state_dict(ckpt["state_dict"])
    ag.eval()
    return ag, c


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    d = ConfigC()
    for k, v in asdict(d).items():
        if not isinstance(v, bool):
            p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    p.add_argument("--no-gae", action="store_true")
    p.add_argument("--no-advnorm", action="store_true")
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="start even if the box is busy. Do not.")
    a = p.parse_args()
    require_quiet_box(a.force, quiet=a.quiet)
    cfg = ConfigC(**{k: getattr(a, k) for k in asdict(d) if hasattr(a, k)})
    cfg.use_gae, cfg.use_advnorm, cfg.use_clip = (not a.no_gae, not a.no_advnorm,
                                                  not a.no_clip)
    print(f"=== ppo-continuous  tag={cfg.tag}  seed={cfg.seed} ===")
    r = train(cfg, verbose=not a.quiet)
    print(f"  steps_to_threshold : {r['steps_to_threshold']}")
    print(f"  final rolling100   : {r['final_rolling100']:.1f}")
    print(f"  eval (mean action) : {r['greedy_eval_mean']:.1f} (min {r['greedy_eval_min']:.0f})")
    print(f"  final sigma        : {r['final_sigma']:.3f}")
    print(f"  action saturation  : {r['saturation_fraction']:.1%} of samples outside [-1,1]")
    print(f"  wall clock         : {r['wall_clock_s']:.1f} s")
