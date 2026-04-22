#!/usr/bin/env python3
"""오프라인 W&B run-*.wandb 에서 마지막 train/global_step · train/loss 등을 출력합니다."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _find_newest_run_file(wandb_dir: Path) -> Path:
    runs = list(wandb_dir.glob("run-*.wandb"))
    if not runs:
        raise FileNotFoundError(f"run-*.wandb 없음: {wandb_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def scan_run_file(run_file: Path) -> dict:
    from wandb.sdk.internal.datastore import DataStore

    ds = DataStore()
    ds.open_for_scan(run_file)
    best: dict = {}
    while True:
        try:
            rec = ds.scan_record()
            if rec is None:
                break
        except Exception:
            break
        if not isinstance(rec, tuple) or len(rec) < 2 or not isinstance(rec[1], bytes):
            continue
        blob = rec[1]
        if b"train/global_step" not in blob:
            continue
        sm = re.search(rb"train/global_step[^\d]{0,8}(\d+)", blob)
        if not sm:
            continue
        step = int(sm.group(1))
        if step < best.get("step", -1):
            continue
        best["step"] = step
        lm = re.search(rb"train/loss[^\d]{0,12}([\d.]+)", blob)
        if lm:
            best["loss"] = float(lm.group(1))
        em = re.search(rb"train/epoch[^\d]{0,12}([\d.]+)", blob)
        if em:
            best["epoch"] = float(em.group(1))
        gm = re.search(rb"train/grad_norm[^\d]{0,16}([\d.]+)", blob)
        if gm:
            best["grad_norm"] = float(gm.group(1))
        rm = re.search(rb"train/learning_rate[^\d]{0,16}([\d.eE+-]+)", blob)
        if rm:
            best["learning_rate"] = float(rm.group(1))
    return best


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_dir = repo_root / "wandb" / "latest-run"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wandb-dir",
        type=Path,
        default=default_dir,
        help=f"run-*.wandb 이 있는 디렉터리 (기본: {default_dir})",
    )
    p.add_argument(
        "--run-file",
        type=Path,
        default=None,
        help="직접 지정할 run-XXXX.wandb 경로 (--wandb-dir 무시)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="한 줄 JSON으로 출력",
    )
    args = p.parse_args()

    try:
        run_file = args.run_file if args.run_file is not None else _find_newest_run_file(args.wandb_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    metrics = scan_run_file(run_file)
    if not metrics:
        print(f"메트릭 없음 (파일: {run_file})", file=sys.stderr)
        return 2

    if args.json:
        out = {"run_file": str(run_file), **metrics}
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"run_file: {run_file}")
        for k in ("step", "loss", "epoch", "learning_rate", "grad_norm"):
            if k in metrics:
                print(f"  {k}: {metrics[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
