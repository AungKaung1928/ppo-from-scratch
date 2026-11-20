"""compute_gae checked by hand. Exits 1 on failure; run as `python test_gae.py`.

Convention in ppo.py: dones[t] = 1 means step t STARTS a new episode, so the
bootstrap from step t-1 into step t is cut (last_done plays the same role for
the step after the batch). Episode boundaries below: A = steps 0-1
(terminated), B = step 2 (truncated: the caller already folded
gamma * V(terminal) = 0.9 * 2.0 = 1.8 into r[2]), C = step 3, still running.
"""
import sys

import numpy as np
import torch

from ppo import compute_gae


def check(name, got, want):
    ok = np.allclose(got, want, atol=1e-6)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {np.round(np.asarray(got, dtype=np.float64), 6).tolist()}, want {np.round(want, 6).tolist()}")
    return ok


def main():
    gamma, lam = 0.9, 0.8
    r = torch.tensor([[1.0], [2.0], [0.5 + 1.8], [1.0]])
    v = torch.tensor([[0.5], [1.0], [0.7], [0.3]])
    d = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    adv, ret = compute_gae(r, v, d, torch.tensor([0.4]), torch.tensor([0.0]), gamma, lam)
    # backward by hand
    # t=3: 1.0 + 0.9*0.4 - 0.3 = 1.06
    # t=2: step 3 starts C -> no bootstrap: 2.3 - 0.7 = 1.6
    # t=1: step 2 starts B -> 2.0 - 1.0 = 1.0
    # t=0: delta 1.0 + 0.9*1.0 - 0.5 = 1.4; A = 1.4 + 0.9*0.8*1.0 = 2.12
    want = np.array([2.12, 1.0, 1.6, 1.06])
    ok = check("advantages, 1 termination + 1 truncation", adv[:, 0].numpy(), want)
    ok &= check("returns = advantages + values", ret[:, 0].numpy(), want + v[:, 0].numpy())

    # lambda = 1, no boundaries: advantage = discounted return - V
    g = torch.Generator().manual_seed(0)
    r = torch.rand(6, 2, generator=g)
    v = torch.rand(6, 2, generator=g)
    last = torch.rand(2, generator=g)
    adv, _ = compute_gae(r, v, torch.zeros(6, 2), last, torch.zeros(2), 0.95, 1.0)
    want = np.zeros((6, 2))
    acc = last.numpy().astype(np.float64)
    for t in reversed(range(6)):
        acc = r[t].numpy() + 0.95 * acc
        want[t] = acc - v[t].numpy()
    ok &= check("lambda=1 equals discounted return minus V", adv.numpy().ravel(), want.ravel())

    # last_done = 1: no bootstrap past the batch end
    adv, _ = compute_gae(torch.tensor([[1.0]]), torch.tensor([[0.2]]), torch.tensor([[0.0]]),
                         torch.tensor([5.0]), torch.tensor([1.0]), 0.9, 0.8)
    ok &= check("last_done cuts the final bootstrap", adv[:, 0].numpy(), np.array([0.8]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
