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
# docs/DETAILS.md holds every table; the short README.md repeats a few of its rows
with open(os.path.join(HERE, "docs", "DETAILS.md"), encoding="utf-8") as f:
    LINES = f.read().splitlines()
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    SHORT = f.read()

study = {r["tag"]: r for r in S["study"]}
sig = {r["tag"]: r for r in S["significance"]}


def line(prefix, metric):
    for l in LINES:
        if l.startswith(f"| {prefix} | {metric}"):
            return l
    raise AssertionError(f"README row not found: {prefix} | {metric}")


for l in SHORT.splitlines():
    if l.startswith("| ") and not l.startswith("| method |"):
        check(f"short row {l.split('|')[1].strip()}", l in LINES, l)
for n in ("62,144", "500.0", "2.34x"):
    check(f"short README number {n}", n in SHORT, n)

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

# step-5 ablation table: every cell of every row (bold stripped)
STEP5 = {l.replace("**", "") for l in LINES}
for tag, label in (("baseline", "baseline"), ("no_advnorm", "no advantage normalisation"),
                   ("no_gae", "no GAE (lambda = 1)"), ("no_clip", "no ratio clipping")):
    r = study[tag]
    tail = "1.00x | — |" if tag == "baseline" else f"{sig[tag]['ratio']:.2f}x | " + (
        f"{sig[tag]['p']:.3f} |" if tag == "no_advnorm" else f"{sig[tag]['p']:.4f} |")
    row5 = f"| {label} | {r['solved']}/{r['n']} | {r['stt_median']:,.0f} | {tail}"
    check(f"step-5 table row {tag}", row5 in STEP5, row5)

# step 6, continuous actions: rows vs runs_c/summary.json
with open(os.path.join(HERE, "runs_c", "summary.json")) as f:
    SC = json.load(f)
cstudy = {r["tag"]: r for r in SC["study"]}
csig = {r["tag"]: r for r in SC["significance"]}
cb = cstudy["baseline"]
cost = cb["stt_median"] / b["stt_median"] - 1
l = line("PPO continuous (Gaussian)", "steps-to-threshold")
check("continuous row", f"**{cb['stt_median']:,.0f}** | IQR [{cb['stt_q1']:,.0f}, {cb['stt_q3']:,.0f}], "
      f"{cb['solved']}/{cb['n']}, {100 * cost:.1f}% over discrete" in l, l)
l = line("**continuous, no ratio clipping**", "solved")
cn = cstudy["no_clip"]
check("continuous no_clip row", f"**{cn['solved']}/{cn['n']}**" in l and f"median greedy return {cn['greedy_median']:.0f}" in l, l)
l = line("**continuous, `ppo_continuous.py`**", "")
check("continuous step-6 table", f"**{cb['solved']}/{cb['n']}** | **{cb['stt_median']:,.0f}** | "
      f"**[{cb['stt_q1']:,.0f}, {cb['stt_q3']:,.0f}]** | **{cb['greedy_median']:.1f}**" in l, l)
# the continuous ablation table is the one with sigma and saturation columns; its rows reuse the step-5 labels
top = LINES.index("| ablation | solved | steps median | vs baseline | permutation p | final sigma | saturation |")
CTAB = {l.split(" | ")[0][2:]: l for l in LINES[top + 2:top + 6]}
for tag, label in (("no_advnorm", "no advantage normalisation"), ("no_gae", "no GAE (lambda = 1)")):
    r, s = cstudy[tag], csig[tag]
    l = CTAB[label]
    check(f"continuous {tag} ablation", f"| {r['solved']}/{r['n']} | {r['stt_median']:,.0f} | {s['ratio']:.2f}x |" in l, l)
l = CTAB["**no ratio clipping**"]
check("continuous no_clip ablation", f"**{csig['no_clip']['ratio']:.2f}x** | {csig['no_clip']['p']:.4f}" in l, l)
for tag, label in (("baseline", "baseline"), ("no_advnorm", "no advantage normalisation"),
                   ("no_gae", "no GAE (lambda = 1)"), ("no_clip", "**no ratio clipping**")):
    r = cstudy[tag]
    sat = f"{100 * r['saturation_median']:.1f}%"
    check(f"continuous {tag} sigma + saturation", CTAB[label].endswith(
        f"| {r['final_sigma_median']:.3f} | {'**' + sat + '**' if tag == 'no_clip' else sat} |"), CTAB[label])
check("no_clip saturation max", f"up to {100 * cn['saturation_max']:.1f}%" in "\n".join(LINES))
check("continuous cost sentence", f"costs {100 * cost:.1f}% of" in "\n".join(LINES))
# remaining cells of the step-6 tables: discrete row, baseline/p/no_clip cells, side-by-side table
row = (f"| discrete, `ppo.py` (step 4) | {b['solved']}/{b['n']} | {b['stt_median']:,.0f} | "
       f"[{b['stt_q1']:,.0f}, {b['stt_q3']:,.0f}] | {b['greedy_median']:.1f} |")
check("step-6 discrete row", row in LINES, row)
check("continuous baseline ablation row", CTAB["baseline"].startswith(f"| baseline | {cb['solved']}/{cb['n']} | {cb['stt_median']:,.0f} | 1.00x | — |"))
check("continuous p-values", f"| **{csig['no_advnorm']['p']:.2f}** |" in CTAB["no advantage normalisation"]
      and f"| {csig['no_gae']['p']:.4f} |" in CTAB["no GAE (lambda = 1)"])
check("continuous no_clip cells", CTAB["**no ratio clipping**"].startswith(
    f"| **no ratio clipping** | **{cn['solved']}/{cn['n']}** | **>150,000** |") and cn["stt_median"] is None)
for label, tag, dp, cp in (("no advantage normalisation", "no_advnorm", "{:.3f}", "{:.2f}"), ("no GAE", "no_gae", "{:.4f}", "{:.4f}")):
    row = (f"| {label} | {sig[tag]['ratio']:.2f}x, {study[tag]['solved']}/{study[tag]['n']}, p={dp.format(sig[tag]['p'])} | "
           f"{csig[tag]['ratio']:.2f}x, {cstudy[tag]['solved']}/{cstudy[tag]['n']}, p={cp.format(csig[tag]['p'])} |")
    check(f"side-by-side {tag}", row in LINES, row)
row = (f"| **no ratio clipping** | **{sig['no_clip']['ratio']:.2f}x, {study['no_clip']['solved']}/{study['no_clip']['n']} solved** | "
       f"**{csig['no_clip']['ratio']:.2f}x, {cn['solved']}/{cn['n']} solved** |")
check("side-by-side no_clip", row in LINES, row)

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
# step-6 comparison tables: basin per controller, failures per init range (LQR | PPO median | PPO worst)
check("basin table theta", f"| critical `theta_dot0` | **{t[0]:.3f} rad/s** | {t[1]:.3f} rad/s | {t[2]:.3f} rad/s |" in LINES)
check("basin table x", f"| critical `x_dot0` | **{x[0]:.3f} m/s** | {x[1]:.3f} m/s | {x[2]:.3f} m/s |" in LINES)
for k, wk in C["widened"].items():
    row6 = f"| U(-{k}, {k}) | **{wk['lqr']}** | {median(wk['ppo']):.0f} | {max(wk['ppo'])} |"
    check(f"widened table {k}", row6 in LINES, row6)
row6 = (f"| nominal return, 100 episodes | {C['nominal_lqr']:.1f} | {median(C['nominal_ppo']):.1f} | "
        f"{max(C['nominal_ppo']):.1f} |")
check("comparison nominal row", row6 in LINES, row6)


# "8 seeds vs 16 seeds" table: ratio of medians over study seeds 0-7 and 0-15, from the raw runs/study/*.json
def study_median(tag, n):
    v = []
    for s in range(n):
        with open(os.path.join(HERE, "runs", "study", f"{tag}_seed{s}.json")) as f:
            r = json.load(f)
        v.append(r["steps_to_threshold"] if r["solved"] else r["config"]["total_timesteps"] + 1)
    return median(v)


for tag, label in (("no_gae", "no GAE"), ("no_advnorm", "no advantage normalisation")):
    r8, r16 = (study_median(tag, n) / study_median("baseline", n) for n in (8, 16))
    row8 = f"| {label} | {r8:.2f}x | **{r16:.2f}x** |"
    check(f"8 vs 16 seeds row {tag}", row8 in LINES and f"{r16:.2f}" == f"{sig[tag]['ratio']:.2f}", row8)

# runs/summary.json and runs_c/summary.json themselves: per-ablation stats are exactly study.summarise() over the raw runs
import numpy as np  # noqa: E402


def check_summary_from_raw(runs, rows):
    for tag, row in rows.items():
        rs = []
        for s in range(row["n"]):
            with open(os.path.join(HERE, runs, "study", f"{tag}_seed{s}.json")) as f:
                rs.append(json.load(f))
        cap = rs[0]["config"]["total_timesteps"]
        q1, med, q3 = np.percentile([r["steps_to_threshold"] if r["solved"] else cap + 1 for r in rs], [25, 50, 75])
        calc = {"solved": sum(r["solved"] for r in rs), "stt_median": None if med > cap else float(med),
                "stt_q1": None if q1 > cap else float(q1), "stt_q3": None if q3 > cap else float(q3),
                "greedy_median": float(np.median([r["greedy_eval_mean"] for r in rs])),
                "greedy_min": float(min(r["greedy_eval_mean"] for r in rs))}
        if "saturation_median" in row:      # continuous family only
            calc.update(final_sigma_median=float(np.median([r["final_sigma"] for r in rs])),
                        saturation_median=float(np.median([r["saturation_fraction"] for r in rs])),
                        saturation_max=float(max(r["saturation_fraction"] for r in rs)))
        check(f"{runs}/summary.json {tag} == raw runs", all(calc[k] == row[k] for k in calc),
              {k: (calc[k], row[k]) for k in calc if calc[k] != row[k]})


check_summary_from_raw("runs", study)
check_summary_from_raw("runs_c", cstudy)
check("theta span sentence", f"spans {t[3]:.3f} (seed 5) to {t[2]:.3f} (seed 0)" in "\n".join(LINES))

# LQR vs PPO summary table: the cells the rows above do not reach
with open(os.path.join(HERE, "lqr_baseline.json"), encoding="utf-8") as f:
    LQ = json.load(f)
l = line("samples spent on the design", "")
check("summary samples row", f"**{LQ['sample_cost_env_steps']} environment steps** | **{b['stt_median']:,.0f}** steps-to-threshold, "
      f"median, IQR [{b['stt_q1']:,.0f}, {b['stt_q3']:,.0f}]" in l, l)
l = line("nominal return, 100 consecutive episodes", "")
check("summary nominal row", f"**{LQ['mean_return_100']:.1f}** (1000 episodes: {LQ['mean_return_1000']:.2f}, 0 failures) | "
      f"**{b['greedy_median']:.1f}** median, {b['greedy_min']:.1f} worst seed" in l and LQ["mean_return_1000"] == 500.0, l)
l = line("spread across 16 seeds", "")
check("summary spread row", f"{b['solved']}/{b['n']} solved, yet" in l, l)
l = line("moved to the MuJoCo plant (step 6)", "")
check("summary MuJoCo row (PPO cell)", f"{cb['stt_median']:,.0f} median, {cb['solved']}/{cb['n']}" in l, l)

# step-4 "baseline PPO vs LQR" table
for row4 in (f"| solved (>=475 over 100 consecutive episodes) | **{b['solved']}/{b['n']}** | yes |",
             f"| steps-to-threshold, median | **{b['stt_median']:,.0f}** | **{LQ['steps_to_threshold']}** |",
             f"| steps-to-threshold, IQR | [{b['stt_q1']:,.0f}, {b['stt_q3']:,.0f}] | — |",
             f"| greedy eval, 100 episodes, median | **{b['greedy_median']:.1f}** | {LQ['mean_return_100']:.1f} |",
             f"| greedy eval, worst seed | {b['greedy_min']:.1f} | — |"):
    check(f"step-4 table {row4.split('|')[1].strip()}", row4 in LINES, row4)

# short README prose: the same numbers sit in its table too, so pin each sentence's own line
(head,) = [x for x in SHORT.splitlines() if x.startswith("PPO written from scratch")]
check("short README headline", f"both reach {LQ['mean_return_100']:.1f}, but PPO spends {b['stt_median']:,.0f} environment steps "
      f"(median, {b['n']} seeds) and its recovery basin is half the LQR's" in head and 0.45 < t[1] / t[0] < 0.55, head)
(clip,) = [x for x in SHORT.splitlines() if x.startswith("Only ratio clipping matters")]
check("short README clipping sentence", f"({sig['no_clip']['ratio']:.2f}x without it)" in clip, clip)

# step-2 LQR tables vs lqr_baseline.json (results, basin per controller, Q sensitivity) and compare.json (widened)
for metric, cell in (("mean return, 100 consecutive episodes", f"**{LQ['mean_return_100']:.1f}**"),
                     ("mean return, 1000 episodes", f"**{LQ['mean_return_1000']:.2f}**, 0 failures"),
                     ("sample cost of the design", f"**{LQ['sample_cost_env_steps']} environment steps**"),
                     ("steps-to-threshold", f"**{LQ['steps_to_threshold']}**")):
    check(f"LQR result row: {metric}", f"| {metric} | {cell} |" in LINES, cell)
for key, unit in (("theta_dot0", "rad/s"), ("x_dot0", "m/s")):
    r = LQ["basin"][key]
    cell = f"| critical `{key}` | **{r['bangbang']:.4f} {unit}** | {r['continuous_clipped']:.4f} {unit} |"
    check(f"LQR basin row {key}", cell in LINES, cell)
for label, key in (("`I4`", "I4"), ("`diag(1,1,10,10)` pole-weighted", "diag(1,1,10,10)"),
                   ("`diag(10,10,1,1)` cart-weighted", "diag(10,10,1,1)"), ("`diag(0,0,1,1)` pole only", "diag(0,0,1,1) pole")):
    q = LQ["q_sensitivity"][key]
    l = line(label, "")
    cells = [c.strip().strip("*") for c in l.split("|")[3:7]]
    check(f"Q sensitivity row {key}", cells == [f"{q['rho']:.4f}", f"{q['mean_return']:.1f}", f"{q['crit_theta_dot']:.2f} rad/s",
                                                f"{q['max_abs_x']:.3f} m"], l)
for k in ("1.0", "1.5", "2.0", "2.5"):
    l = line(f"U(-{k}, {k})", "")
    check(f"LQR widened {k} failures", l.endswith(f"| {C['widened'][k]['lqr']} |"), l)

# hyperparameter search table vs runs/hpsearch/*.json, same statistic as study.summarise (unsolved = budget + 1)
import glob

import numpy as np

HP = {}
for p in glob.glob(os.path.join(HERE, "runs", "hpsearch", "*.json")):
    with open(p) as f:
        r = json.load(f)
    HP.setdefault(r["config"]["tag"], []).append(r)
for label, tag in (("`lr 2e-3`, 10 epochs", "lr2e3_ep10"), ("`lr 1e-3`, 8 epochs, 8 minibatches", "lr1e3_ep8mb8"),
                   ("`lr 1.5e-3`, 10 epochs", "lr15e4_ep10"), ("`lr 1e-3`, 10 epochs", "lr1e3_ep10"),
                   ("`lr 2.5e-3`, 4 epochs", "lr25e4_ep4"), ("`lr 3e-4`, 10 epochs", "lr3e4_ep10"),
                   ("`lr 1e-3`, 4 epochs", "lr1e3_ep4"), ("`lr 3e-4`, 4 epochs", "lr3e4_ep4")):
    rs = HP[tag]
    budget = rs[0]["config"]["total_timesteps"]
    q1, med, q3 = np.percentile([r["steps_to_threshold"] if r["solved"] else budget + 1 for r in rs], [25, 50, 75])
    f = lambda x: f">{budget:,}" if x > budget else f"{x:,.0f}"
    tail = f"| {sum(r['solved'] for r in rs)}/{len(rs)} | {f(med)} | " + ("—" if q3 > budget else f"[{f(q1)}, {f(q3)}]") + " |"
    rows = [l.replace("**", "") for l in LINES if l.replace("**", "").startswith(f"| {label}")]
    check(f"hp search row {tag}", len(rows) == 1 and rows[0].endswith(tail), (rows, tail))

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


def lrow(i):
    return next(l for l in LINES if l.startswith(f"| L{i} | "))


def lnum(pattern):
    return float(re.search(pattern, out).group(1))


# L1-L7 table vs the same lqr.py run: roundoff values by order (they move across CPUs), the rest exact
for i, pat, tol in ((1, r"L1\..*max abs diff ([0-9.e+-]+)", 1e-9), (2, r"L2\..*max abs diff ([0-9.e+-]+)", 1e-12),
                    (5, r"L5\..*max rel err ([0-9.e+-]+)", 1e-9), (6, r"L6\..*max rel diff ([0-9.e+-]+)", 1e-12)):
    doc = float(re.search(r"\*\*([0-9.e-]+)\*\*", lrow(i)).group(1))
    check(f"L{i} at roundoff", lnum(pat) < tol and doc < tol, (lnum(pat), doc))
l3 = re.search(r"residual ([0-9.e+-]+), (\d+) sweeps", out)
check("L3 row", f"residual **{float(l3.group(1)):.2e}**, {l3.group(2)} sweeps |" in lrow(3), l3.group(0))
check("L4 row", f"| **{lnum(r'rho = ([0-9.]+)'):.6f}** |" in lrow(4))
l7 = re.search(r"L7\..*?([0-9.]+) \(min (\d+), max (\d+)\)", out)
check("L7 row", f"**{l7.group(1)}** (min {l7.group(2)}, max {l7.group(3)})" in lrow(7), l7.group(0))

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


def near(s, v):
    """README string s is v rounded to s's last digit (v itself comes printed to 3-4 significant digits)."""
    m, _, e = s.partition("e")
    step = 10.0 ** ((int(e) if e else 0) - (len(m.split(".")[1]) if "." in m else 0))
    return abs(float(s) - v) <= 0.51 * step


def ve_num(pattern):
    return float(re.search(pattern, ve).group(1))


def cell(prefix, pattern, metric=""):
    return re.search(pattern, line(prefix, metric)).group(1)


# verification table: seeded numbers exactly, roundoff numbers by order (they move across CPUs, see check 1)
check("check 2a drift at roundoff", ve_num(r"max relative drift ([0-9.e+-]+)") < 1e-12
      and float(cell("2a", r"\*\*([0-9.e-]+)\*\*", "Energy conserved")) < 1e-12)
check("check 2b drift at roundoff", ve_num(r"max absolute drift ([0-9.e+-]+)") < 1e-12
      and float(cell("2b", r"\*\*([0-9.e-]+)\*\*", "Horizontal momentum")) < 1e-12)
check("check 3 rel err", ve_num(r"rel err ([0-9.e+-]+) \(tol") < 1e-9 and float(cell("3", r"rel err \*\*([0-9.e-]+)\*\*", "Pinned cart")) < 1e-9)
steps = re.search(r"terminated after (\d+) steps, return ([0-9.]+)", ve)
check("check 4c row", f"always-right policy: {steps.group(1)} steps, return {steps.group(2)}" in line("4c", "Reward is 1.0"), steps.group(0))
thr = re.search(r"theta_threshold ([0-9.]+) rad", ve).group(1)
check("check 4d row", f"| {thr} rad |" in line("4d", "Termination bounds"), thr)
check("check 4e row", f"**{ve_num(r'mean return ([0-9.]+) over 100 episodes'):.1f} / 500** over 100 episodes" in line("4e", "Action sign"))
rnd = re.search(r"(\d+) episodes: mean ([0-9.]+) \+/- ([0-9.]+)", ve)
check("check 6 row", f"**{rnd.group(2)} ± {rnd.group(3)}** over {int(rnd.group(1)) // 1000}k episodes" in line("6", "Random-policy"), rnd.group(0))
n_golden = re.search(r"(\d+) states, max abs diff", ve).group(1)
check("check 7 row", f"bitwise identical, {n_golden} states" in line("7", "Golden"))

# measured-properties table (deterministic; throughput is machine-dependent and not checked)
lam_v = float(lam.group(1))
check("instability rate + e-fold", near(cell("Instability rate about upright, pinned cart", r"\*\*([0-9.]+) s"), lam_v)
      and near(cell("Instability rate about upright, pinned cart", r"e-fold in (\d+) ms"), 1000 / lam_v))
eu = re.search(r"median \|d theta\| ([0-9.e+-]+) rad, max ([0-9.e+-]+) rad \(([0-9.]+)% of", ve)
l = line("One-step Euler error vs RK4, in the operational envelope", "")
got = re.search(r"median \*\*([0-9.e-]+) rad\*\*, max ([0-9.e-]+) rad \(([0-9.]+)% of", l)
check("one-step Euler row", all(near(got.group(i), float(eu.group(i))) for i in (1, 2, 3)), (eu.group(0), l))
e25 = re.search(r"\|d theta\| ([0-9.e+-]+) rad \(([0-9.]+)% of the limit\), \|d x\|", ve)
got = re.search(r"\*\*([0-9.e-]+) rad\*\* \(([0-9.]+)% of the limit\)", line("Euler vs RK4 after 25 controlled steps (0.5 s), identical forces", ""))
check("25-step Euler row", all(near(got.group(i), float(e25.group(i))) for i in (1, 2)), e25.group(0))
check("Euler energy gain row", near(cell("Explicit-Euler energy gain, F=0, 50 steps", r"\*\*\+([0-9.]+)%\*\*"), ve_num(r"\+([0-9.]+)% after 50 steps")))
got = re.search(r"\| \*\*\+([0-9.]+)%\*\* \| \+([0-9.]+)% \|", line("our explicit Euler (step 1)", ""))
check("V3 explicit Euler row", near(got.group(1), ve_num(r"\+([0-9.]+)% after 50 steps"))
      and near(got.group(2), ve_num(r"\+([0-9.]+)% after 500")))

# step-6 MuJoCo checks (V1-V5) vs a fresh mj_cartpole.py run; mujoco is an optional dependency (CI installs it)
try:
    import mujoco  # noqa: F401
except ImportError:
    print("  skip  MuJoCo rows: mujoco not installed")
else:
    print("MuJoCo-plant numbers vs a fresh mj_cartpole.py run (~3 s)")
    mj = subprocess.run([sys.executable, os.path.join(HERE, "mj_cartpole.py")], capture_output=True, text=True, check=True).stdout
    v1 = re.search(r"max abs ([0-9.e+-]+), max rel ([0-9.e+-]+)", mj)
    got = re.search(r"max abs \*\*([0-9.e-]+)\*\*, max rel \*\*([0-9.e-]+)\*\*", line("V1. MuJoCo `qacc` == `cartpole.accelerations()`, 5000 random states", ""))
    check("V1 at roundoff", all(float(v1.group(i)) < 1e-11 and float(got.group(i)) < 1e-11 for i in (1, 2)), v1.group(0))
    v2 = re.search(r"median \|dtheta\| ([0-9.e+-]+) rad, max ([0-9.e+-]+) \(([0-9.]+)% of the limit\)", mj)
    got = re.search(r"median \\\|dθ\\\| \*\*([0-9.e-]+) rad\*\*, max \*\*([0-9.e-]+)\*\* \(([0-9.]+)% of",
                    line("V2. one-step gap, same state and same force", ""))
    check("V2 one-step gap row", all(near(got.group(i), float(v2.group(i))) for i in (1, 2, 3)), v2.group(0))
    v3 = re.search(r"\+([0-9.]+)% at 50 steps, \+([0-9.]+)% at 500", mj)
    got = re.search(r"\| \+([0-9.]+)% \| \*\*\+([0-9.]+)%\*\* \|", line("MuJoCo semi-implicit", ""))
    check("V3 MuJoCo row", all(near(got.group(i), float(v3.group(i))) for i in (1, 2)), v3.group(0))
    v5 = re.findall(r"V5\. step-2 LQR gain on the MuJoCo plant, (bang-bang|continuous).*?mean ([0-9.]+) over 100 episodes \(min (\d+)\)", mj)
    for name, m, lo in v5:
        label = "bang-bang, sign only, ±10 N" if name == "bang-bang" else "continuous, clipped to ±10 N"
        check(f"V5 {name} row", f"| {label} | **{float(m):.1f}** (min {lo}) |" in LINES, (m, lo))
    check("V5 both controllers", len(v5) == 2, str(v5))
    (m5, lo5), = {(float(m), lo) for _, m, lo in v5}        # one Results row speaks for both controllers
    check("Results row step-2 K on MuJoCo", f"| step-2 `K` on the MuJoCo plant | mean return, 100 episodes | **{m5:.1f}** | "
          f"bang-bang and continuous, min {lo5} |" in LINES, (m5, lo5))

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
