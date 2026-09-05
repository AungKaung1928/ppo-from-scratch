"""
Steps 4 and 5 -- the seed study and the ablations.

WHY THIS FILE IS SEPARATE FROM ppo.py
-------------------------------------
A single training run tells you almost nothing about a PPO configuration. The
variance across seeds on cart-pole is larger than most of the differences people
report between algorithms, so one curve is not evidence, it is an anecdote.
Everything reported from here is >= 5 seeds with a median and an interquartile
range, and the headline statistic is steps-to-threshold, not return.

Return cannot be the headline. Step 2 measured why: on this task return
saturates at 500 for anything competent, and it ranked four visibly different
LQR designs -- including one whose closed loop is only marginally stable -- as
exactly equal. Steps-to-threshold still discriminates.

CENSORING
---------
A run that never reaches threshold inside the step budget has no
steps-to-threshold. Dropping those runs and taking the median of the survivors
is the standard way to make a bad configuration look good, so they are not
dropped: an unsolved run is entered as `budget + 1`, which makes the median
correct as long as fewer than half the runs fail, and reports as ">150000" when
more than half do. The solved count is always printed next to it.

SEEDS
-----
Study seeds are 0-7. The hyperparameter search used 100-103 and nothing else.
Those sets do not overlap, so the reported numbers are not the numbers that were
tuned on.

BOX ETIQUETTE
-------------
This machine shares 8 threads with a detection-training job that has priority.
Runs are launched at nice 10, four at a time, one torch thread each, and the
launcher refuses to start if the 1-minute load average says something else is
already using the machine. Pass --force to override that.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from boxcheck import require_quiet_box
from ppo import Config, train
from ppo_continuous import ConfigC
from ppo_continuous import train as train_c

STUDY_SEEDS = list(range(16))
SEARCH_SEEDS = [100, 101, 102, 103]
MAX_PARALLEL = 4
LOAD_LIMIT = 4.0

ABLATIONS = {
    "baseline":     {},
    "no_gae":       {"use_gae": False},
    "no_advnorm":   {"use_advnorm": False},
    "no_clip":      {"use_clip": False},
}

SEARCH = {
    "lr3e4_ep4":    {"lr": 3e-4, "update_epochs": 4},     # the first thing I wrote
    "lr1e3_ep4":    {"lr": 1e-3, "update_epochs": 4},
    "lr3e4_ep10":   {"lr": 3e-4, "update_epochs": 10},
    "lr1e3_ep10":   {"lr": 1e-3, "update_epochs": 10},
    "lr15e4_ep10":  {"lr": 1.5e-3, "update_epochs": 10},
    "lr2e3_ep10":   {"lr": 2e-3, "update_epochs": 10},    # adopted
    "lr1e3_ep8mb8": {"lr": 1e-3, "update_epochs": 8, "num_minibatches": 8},
    "lr25e4_ep4":   {"lr": 2.5e-3, "update_epochs": 4},
}


def _run(cfg):
    return (train_c if isinstance(cfg, ConfigC) else train)(cfg, verbose=False)


def run_grid(spec, seeds, out, base=None):
    base = base if base is not None else Config()
    jobs = [replace(base, seed=s, tag=tag, out=out, **kw)
            for tag, kw in spec.items() for s in seeds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        results = list(ex.map(_run, jobs))
    print(f"  {len(jobs)} runs in {time.perf_counter() - t0:.1f} s wall "
          f"({MAX_PARALLEL}-way parallel, nice 10)")
    return results


def summarise(results, budget):
    """Median and IQR of steps-to-threshold, unsolved runs entered as censored."""
    by = {}
    for r in results:
        by.setdefault(r["config"]["tag"], []).append(r)
    rows = []
    for tag, rs in by.items():
        rs.sort(key=lambda r: r["config"]["seed"])
        stt = np.array([r["steps_to_threshold"] if r["solved"] else budget + 1
                        for r in rs], dtype=float)
        q1, med, q3 = np.percentile(stt, [25, 50, 75])
        greedy = np.array([r["greedy_eval_mean"] for r in rs])
        rows.append({
            "tag": tag,
            "n": len(rs),
            "solved": int(sum(r["solved"] for r in rs)),
            "stt_median": None if med > budget else float(med),
            "stt_q1": None if q1 > budget else float(q1),
            "stt_q3": None if q3 > budget else float(q3),
            "stt_iqr": None if (q3 > budget or q1 > budget) else float(q3 - q1),
            "stt_all": [r["steps_to_threshold"] for r in rs],
            "greedy_median": float(np.median(greedy)),
            "greedy_q1": float(np.percentile(greedy, 25)),
            "greedy_q3": float(np.percentile(greedy, 75)),
            "greedy_min": float(greedy.min()),
            "final_rolling_median": float(np.median([r["final_rolling100"] for r in rs])),
            "wall_median": float(np.median([r["wall_clock_s"] for r in rs])),
        })
    return rows


def fmt(x, budget):
    return f">{budget}" if x is None else f"{int(x)}"


def print_table(rows, budget, title):
    print(f"\n{title}")
    print(f"  {'config':<14} {'solved':>7} {'steps-to-threshold':>28} "
          f"{'greedy eval (100 ep)':>26}")
    print(f"  {'':<14} {'':>7} {'median':>9} {'IQR':>18} "
          f"{'median':>9} {'IQR':>16}")
    for r in sorted(rows, key=lambda r: (r["stt_median"] is None,
                                         r["stt_median"] or 0)):
        iqr = ("--" if r["stt_iqr"] is None
               else f"[{fmt(r['stt_q1'],budget)}, {fmt(r['stt_q3'],budget)}]")
        gi = f"[{r['greedy_q1']:.0f}, {r['greedy_q3']:.0f}]"
        print(f"  {r['tag']:<14} {r['solved']:>3}/{r['n']:<3} "
              f"{fmt(r['stt_median'], budget):>9} {iqr:>18} "
              f"{r['greedy_median']:>9.1f} {gi:>16}")


def permutation_test(a, b, n_resamples=100_000, seed=0):
    """Two-sided permutation test on the difference of medians.

    No scipy, and no t-test either: steps-to-threshold is not normal, it is
    right-skewed and censored at the step budget. A permutation test assumes
    only exchangeability under the null, which is exactly what "the ablation
    changed nothing" means. Exhaustive when the number of splits is small
    enough, Monte-Carlo otherwise.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    n = len(a)
    rng = np.random.default_rng(seed)
    from math import comb
    total = comb(len(pool), n)
    if total <= n_resamples:
        from itertools import combinations
        idx_all = list(combinations(range(len(pool)), n))
        count = 0
        for idx in idx_all:
            m = np.zeros(len(pool), bool)
            m[list(idx)] = True
            count += abs(np.median(pool[m]) - np.median(pool[~m])) >= obs - 1e-12
        return count / total, total, True
    count = 0
    for _ in range(n_resamples):
        p = rng.permutation(pool)
        count += abs(np.median(p[:n]) - np.median(p[n:])) >= obs - 1e-12
    return (count + 1) / (n_resamples + 1), n_resamples, False


def load_runs(path):
    import glob
    return [json.load(open(f)) for f in glob.glob(os.path.join(path, "*.json"))]


def significance(results, budget):
    by = {}
    for r in results:
        by.setdefault(r["config"]["tag"], []).append(
            r["steps_to_threshold"] if r["solved"] else budget + 1)
    if "baseline" not in by:
        return []
    base = by["baseline"]
    out = []
    print("\n  is the difference real? two-sided permutation test on the median,")
    print("  unsolved runs censored at budget+1, n_resamples = 100000")
    print(f"    {'config':<14} {'median delta':>14} {'ratio':>8} {'p':>10}  {'kind':<10}")
    for tag, vals in by.items():
        if tag == "baseline":
            continue
        p, n, exact = permutation_test(base, vals)
        d = float(np.median(vals) - np.median(base))
        out.append({"tag": tag, "p": p, "median_delta": d,
                    "ratio": float(np.median(vals) / np.median(base)),
                    "exact": exact, "n_resamples": n})
        print(f"    {tag:<14} {d:>+14.0f} {np.median(vals)/np.median(base):>8.2f} "
              f"{p:>10.4f}  {'exact' if exact else 'monte-carlo'}")
    return out


def check_box(force):
    return require_quiet_box(force, limit=LOAD_LIMIT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hpsearch", "seeds", "ablations",
                                       "analyse", "all"], default="all")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--total-timesteps", type=int, default=Config().total_timesteps)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--family", choices=["discrete", "continuous"],
                    default="discrete",
                    help="discrete = ppo.py on the numpy plant (steps 3-5); "
                         "continuous = ppo_continuous.py on the MuJoCo plant "
                         "(step 6). Different plant AND different action space, "
                         "so the two are never pooled.")
    a = ap.parse_args()
    base = replace(ConfigC() if a.family == "continuous" else Config(),
                   total_timesteps=a.total_timesteps)
    budget = base.total_timesteps
    check_box(a.force)
    os.makedirs(a.out, exist_ok=True)
    summary = {}

    if a.mode in ("hpsearch", "all"):
        print(f"\n=== hyperparameter search (seeds {SEARCH_SEEDS}, "
              f"disjoint from the study seeds) ===")
        res = run_grid(SEARCH, SEARCH_SEEDS, os.path.join(a.out, "hpsearch"), base)
        rows = summarise(res, budget)
        print_table(rows, budget, "search results")
        summary["hpsearch"] = rows

    if a.mode == "analyse":
        res = load_runs(os.path.join(a.out, "study"))
        rows = summarise(res, budget)
        print_table(rows, budget, "results (from saved runs)")
        summary["study"] = rows
        summary["significance"] = significance(res, budget)

    if a.mode in ("seeds", "ablations", "all"):
        spec = {"baseline": {}} if a.mode == "seeds" else ABLATIONS
        print(f"\n=== {'seed study' if a.mode=='seeds' else 'seed study + ablations'} "
              f"(seeds {STUDY_SEEDS}) ===")
        res = run_grid(spec, STUDY_SEEDS, os.path.join(a.out, "study"), base)
        rows = summarise(res, budget)
        print_table(rows, budget, "results")
        summary["study"] = rows
        summary["significance"] = significance(res, budget)
        base_row = next((r for r in rows if r["tag"] == "baseline"), None)
        if base_row and base_row["stt_median"]:
            print("\n  relative to baseline (steps-to-threshold, median):")
            for r in sorted(rows, key=lambda r: (r["stt_median"] is None,
                                                 r["stt_median"] or 0)):
                if r["tag"] == "baseline":
                    continue
                if r["stt_median"] is None:
                    print(f"    {r['tag']:<14} never reaches threshold in "
                          f"{budget} steps ({r['solved']}/{r['n']} seeds solved)")
                else:
                    f_ = r["stt_median"] / base_row["stt_median"]
                    print(f"    {r['tag']:<14} {f_:>5.2f}x baseline "
                          f"({r['solved']}/{r['n']} seeds solved)")

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {os.path.join(a.out, 'summary.json')}\n")


if __name__ == "__main__":
    main()
