"""Restart-on-death wrapper for the long training stages.

Why this exists: a probe run was killed mid-suite by the Linux OOM killer
(dmesg: "Out of memory: Killed process ... (python)"). The main process died
while its pt_data_worker children survived as orphans, which looks exactly
like a hang. Over a ~15 h stage-1 run that is a question of when, not if.

Ultralytics checkpoints last.pt every epoch, so a restart costs at most one
epoch. This wrapper relaunches train.py with --resume until it either
finishes cleanly or stops making progress.

    python scripts/train_watchdog.py --stage 1
    python scripts/train_watchdog.py --stage 1 --max-restarts 20

It deliberately does NOT retry on a clean non-zero exit that produced no new
epoch -- that means the run is failing for a real reason (bad args, missing
data), and restarting would just loop.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "repos/fire_detection"
RUNS = ROOT / "runs"
STAGE_RUN = {1: "stage1_fasdd", 2: "stage2_dfire"}


def epochs_done(run_dir: Path) -> int:
    csv = run_dir / "results.csv"
    if not csv.exists():
        return 0
    return len([l for l in csv.read_text().splitlines()[1:] if l.strip()])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--max-restarts", type=int, default=30)
    p.add_argument("--cooldown", type=int, default=30, help="seconds between restarts")
    args, passthrough = p.parse_known_args()

    run_dir = RUNS / STAGE_RUN[args.stage]
    attempt = 0

    while attempt <= args.max_restarts:
        before = epochs_done(run_dir)
        cmd = [sys.executable, "-u", str(ROOT / "scripts/train.py"), "--stage", str(args.stage)]
        if before > 0:  # a previous attempt got somewhere -- pick up from last.pt
            cmd.append("--resume")
        cmd += passthrough

        print(f"\n=== attempt {attempt} (epochs done: {before}) ===\n{' '.join(cmd)}\n", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        after = epochs_done(run_dir)

        if rc == 0:
            print(f"training finished cleanly after {after} epochs", flush=True)
            return
        if after <= before:
            print(
                f"exit {rc} with no epoch progress ({before} -> {after}). "
                f"This is a real failure, not a crash to retry. Stopping.",
                flush=True,
            )
            sys.exit(rc)

        attempt += 1
        print(f"exit {rc}; progressed {before} -> {after} epochs. "
              f"Restarting in {args.cooldown}s...", flush=True)
        time.sleep(args.cooldown)

    print(f"gave up after {args.max_restarts} restarts", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
