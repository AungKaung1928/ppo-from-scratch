"""README ablation rows against runs/summary.json, basin rows against runs/compare.json (stdlib only).

Steps-to-threshold medians, IQR, solved counts, ratios and permutation p-values in
the results table must equal the study summary they were copied from.

Run:  python test_readme.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


with open(os.path.join(HERE, "runs", "summary.json")) as f:
    S = json.load(f)
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    LINES = f.read().splitlines()

study = {r["tag"]: r for r in S["study"]}
sig = {r["tag"]: r for r in S["significance"]}


def line(prefix, metric):
    for l in LINES:
        if l.startswith(f"| {prefix} | {metric}"):
            return l
    raise AssertionError(f"README row not found: {prefix} | {metric}")


b = study["baseline"]
l = line("PPO baseline (discrete)", "steps-to-threshold")
check("baseline median", f"**{b['stt_median']:,.0f}**" in l, l)
check("baseline IQR", f"IQR [{b['stt_q1']:,.0f}, {b['stt_q3']:,.0f}]" in l)
check("baseline solved", f"{b['solved']}/{b['n']} solved" in l)
l = line("PPO baseline (discrete)", "greedy eval")
check("baseline greedy median = worst seed", f"**{b['greedy_median']:.1f}**" in l and b["greedy_min"] == b["greedy_median"], l)

for tag, label, pfmt in (("no_advnorm", "no advantage normalisation", "permutation p **{:.3f}**"),
                         ("no_gae", "no GAE (lambda = 1)", "p {:.4f}"),
                         ("no_clip", "**no ratio clipping**", "p {:.4f}")):
    r, s = study[tag], sig[tag]
    l = line(label, "steps-to-threshold")
    check(f"{tag} median + ratio", f"{r['stt_median']:,.0f} ({s['ratio']:.2f}x)" in l, l)
    check(f"{tag} solved", f"{r['solved']}/{r['n']}" in l)
    check(f"{tag} p-value", pfmt.format(s["p"]) in l, pfmt.format(s["p"]))

with open(os.path.join(HERE, "runs", "compare.json")) as f:
    C = json.load(f)


def median(xs):
    xs = sorted(xs)
    return (xs[(len(xs) - 1) // 2] + xs[len(xs) // 2]) / 2


th, xd = C["critical"]["theta_dot0"], C["critical"]["x_dot0"]
t = (th["lqr"], median(th["ppo"]), max(th["ppo"]), min(th["ppo"]))
x = (xd["lqr"], median(xd["ppo"]), max(xd["ppo"]))
l = line("basin — critical `theta_dot0` from the origin", "")
check("basin theta headline", f"**{t[0]:.3f} rad/s** | {t[1]:.3f} rad/s median, {t[2]:.3f} best seed, {t[3]:.3f} worst" in l, l)
l = line("basin — critical `x_dot0` from the origin", "")
check("basin x headline", f"**{x[0]:.3f} m/s** | {x[1]:.3f} m/s median, {x[2]:.3f} best seed" in l, l)
l = line("spread across 16 seeds", "")
check("theta spread", f"spans {t[3]:.3f}–{t[2]:.3f} rad/s" in l, l)
l = line("LQR vs PPO basin", "critical `theta_dot0`")
check("basin theta row", f"**{t[0]:.3f}** vs {t[1]:.3f} rad/s | PPO median over 16 seeds, best {t[2]:.3f}" in l, l)
l = line("LQR vs PPO basin", "critical `x_dot0`")
check("basin x row", f"**{x[0]:.3f}** vs {x[1]:.3f} m/s | PPO median over 16 seeds, best {x[2]:.3f}" in l, l)
w = C["widened"]["1.0"]
l = line("failures / 200 episodes, init velocities U(−1.0, 1.0)", "")
check("widened 1.0 failures", f"**{w['lqr']}** | {median(w['ppo']):.0f} median, {max(w['ppo'])} worst seed" in l, l)
check("theta span sentence", f"spans {t[3]:.3f} (seed 5) to {t[2]:.3f} (seed 0)" in "\n".join(LINES))

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
