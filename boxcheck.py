"""
One rule, enforced in one place.

This machine has 8 threads and they are shared with a detection-training job
that has priority. Starting an RL run on top of it corrupts both throughput
measurements and slows down the job that matters more.

study.py had this check from the start. It did not help, because a
hyperparameter sweep launched from a bare shell loop never called study.py --
and that sweep ran straight through a 770%-CPU `train_det.py`. A guard that
only guards the front door is not a guard. So it lives here now and every entry
point that starts training calls it: ppo.py, ppo_continuous.py, study.py.

Learning outcomes are unaffected by contention -- the runs are seeded and
single-threaded, and two identical configs launched under load returned
identical steps-to-threshold. Wall-clock numbers are not, and neither is the
other project's throughput measurement, which is the actual reason for the rule.
"""

import os

LOAD_LIMIT = 4.0


def busy(limit=LOAD_LIMIT):
    return float(open("/proc/loadavg").read().split()[0]) > limit


def require_quiet_box(force=False, limit=LOAD_LIMIT, quiet=False):
    load1 = float(open("/proc/loadavg").read().split()[0])
    if load1 > limit and not force:
        others = ""
        try:
            import subprocess
            out = subprocess.run(["ps", "-eo", "pcpu,args", "--sort=-pcpu"],
                                 capture_output=True, text=True, timeout=5).stdout
            rows = [l for l in out.splitlines()[1:4] if l.strip()]
            others = "\n    " + "\n    ".join(r.strip()[:100] for r in rows)
        except Exception:
            pass
        raise SystemExit(
            f"\n  1-minute load average is {load1:.2f} (limit {limit}).\n"
            f"  Something else is using this machine and it has priority."
            f"{others}\n\n"
            f"  Not starting. Wait for it to finish. --force exists and you "
            f"should not use it.\n")
    if not quiet:
        print(f"  box check: 1-min load {load1:.2f}, ok")
    return load1
