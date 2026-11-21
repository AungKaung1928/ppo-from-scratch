"""README ablation rows vs runs/summary.json, basin rows vs runs/compare.json, LQR and env-check numbers vs fresh lqr.py / verify_env.py runs.

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

print("LQR numbers vs a fresh lqr.py run (numpy, ~5 s)")
import re
import subprocess
import sys
out = subprocess.run([sys.executable, os.path.join(HERE, "lqr.py")], capture_output=True, text=True, check=True).stdout
sweeps = re.search(r"residual ([0-9.e+-]+), (\d+) sweeps", out)
pred = re.search(r"~(\d+) to reach 1e-13 relative \(rho\^2 = ([0-9.]+)\)", out)
rot = re.search(r"max direction diff ([0-9.]+) for R", out)
text = "\n".join(LINES)
check("LQR sweeps + theory", f"**{sweeps.group(2)}** (theory ~{pred.group(1)})" in text, sweeps.group(2))
check("LQR DARE residual", f"DARE residual {float(sweeps.group(1)):.2e}" in text, sweeps.group(1))
check("LQR rho^2 sentence", f"`rho(Ad-BdK)^2 = {pred.group(2)}`" in text and f"**~{pred.group(1)}** sweeps" in text)
check("R rotates gain direction", f"rotates the normalised gain direction by **{rot.group(1)}**" in text, rot.group(1))

print("env-check numbers vs a fresh verify_env.py run (numpy, ~6 s)")
ve = subprocess.run([sys.executable, os.path.join(HERE, "verify_env.py")], capture_output=True, text=True, check=True).stdout
rel = re.search(r"max rel diff ([0-9.e+-]+) \(tol", ve)
lam = re.search(r"lambda measured ([0-9.]+) s\^-1 vs closed form ([0-9.]+) s\^-1", ve)
n_pass = ve.count("[PASS]")
# roundoff differs across CPUs / numpy builds (2.6e-15 here, 2.2e-15 on CI): pin the order, not the digits
r1 = re.search(r"\*\*([0-9.]+e-\d+)\*\*", line("hand-written env, check 1", ""))
check("check 1 rel diff at roundoff", r1 is not None and float(rel.group(1)) < 1e-14 and float(r1.group(1)) < 1e-14,
      f"{rel.group(1)} (README {r1.group(1) if r1 else None})")
check("check 3 divergence rate", f"**{lam.group(1)}** vs {lam.group(2)}" in line("hand-written env, check 3", ""), lam.group(0))
check("13/13 checks", n_pass == 13 and "**done, 13/13 checks pass**" in text, str(n_pass))

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
