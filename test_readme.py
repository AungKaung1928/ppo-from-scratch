"""README ablation rows against runs/summary.json (stdlib only).

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

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
