"""Lead figure for the README: steps-to-threshold per ablation, from runs/summary.json.

    python plot_summary.py            # -> docs/ablation.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {"baseline": "PPO baseline", "no_advnorm": "no advantage norm.",
          "no_gae": "no GAE (lambda = 1)", "no_clip": "no ratio clipping"}


def main():
    with open(os.path.join(HERE, "runs", "summary.json")) as f:
        s = json.load(f)
    rows = sorted(s["study"], key=lambda r: r["stt_median"])
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
    for y, r in enumerate(rows):
        lo, q3 = r["stt_q1"] / 1e3, r["stt_q3"]
        if q3 is None:  # fewer than 3/4 of the seeds solved: the upper quartile never reached the threshold
            ax.plot([lo, 200], [y, y], color="#9aa4b2", lw=2, ls="--")
            hi, note = 200, "upper quartile unsolved"
        else:
            hi, note = q3 / 1e3, ""
            ax.plot([lo, hi], [y, y], color="#9aa4b2", lw=6, solid_capstyle="butt")
        ax.plot(r["stt_median"] / 1e3, y, "o", color="#1f5fbf" if r["tag"] == "baseline" else "#3b3f45", ms=7)
        text = f"{r['stt_median'] / 1e3:,.1f}k, {r['solved']}/{r['n']} solved" + (f", {note}" if note else "")
        ax.text(r["stt_median"] / 1e3, y - 0.32, text, ha="center", va="bottom", fontsize=8)
    ax.set_yticks(range(len(rows)), [LABELS[r["tag"]] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("environment steps to reach the return threshold (thousands), median and IQR, 16 seeds", fontsize=8)
    ax.set_xlim(0, 200)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "docs", "ablation.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
