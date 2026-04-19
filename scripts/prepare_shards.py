"""Download WOMD v1.3.1 scenario tfrecords, convert to shards, delete tfrecord.

Two decoupled thread pools overlap IO and CPU:

* ``dl_workers`` threads each pull a job from ``jobs_q`` and call ``gsutil cp``.
  The downloaded path is pushed onto ``ready_q``. ``ready_q`` is bounded to
  prevent unbounded tfrecord accumulation on disk (backpressure).
* ``cv_workers`` threads each pull from ``ready_q`` and spawn
  ``python -m data.convert`` as a subprocess. After success/failure the
  tfrecord is deleted.

Resume-safe: jobs whose output shard already exists are skipped at plan time.

Usage::

    python -m scripts.prepare_shards training validation testing \\
        [--shard-root DIR] [--tmp DIR] [--dl-workers N] [--cv-workers N]
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

BUCKET = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"

# split name in bucket → (total, shard-prefix, output-subdir)
SPLITS: dict[str, tuple[int, str, str]] = {
    "training":   (1000, "train", "training"),
    "validation": (150,  "val",   "validation"),
    "testing":    (150,  "test",  "test"),
}

REPO = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gen_jobs(splits: list[str], root: Path, tmp_root: Path) -> list[tuple]:
    jobs: list[tuple] = []
    for split in splits:
        total, prefix, outsub = SPLITS[split]
        outdir = root / outsub
        outdir.mkdir(parents=True, exist_ok=True)
        pad_total = f"{total:05d}"
        for i in range(total):
            out = outdir / f"{prefix}.{i}of{total}.pt.zst"
            if out.exists():
                continue
            pad_src = f"{i:05d}"
            src = f"{BUCKET}/{split}/{split}.tfrecord-{pad_src}-of-{pad_total}"
            tmp = tmp_root / f"{split}.tfrecord-{pad_src}"
            jobs.append((split, i, total, src, tmp, out))
    return jobs


def download(job: tuple) -> bool:
    split, i, total, src, tmp, _out = job
    r = subprocess.run(
        ["gsutil", "-q", "cp", src, str(tmp)],
        capture_output=True,
    )
    if r.returncode != 0:
        log(f"[{split} {i}/{total}] DOWNLOAD FAILED: {r.stderr.decode()[-200:]}")
        Path(tmp).unlink(missing_ok=True)
        return False
    log(f"[{split} {i}/{total}] downloaded")
    return True


def convert(job: tuple, min_future_valid: int = 4) -> None:
    split, i, total, _src, tmp, out = job
    r = subprocess.run(
        [sys.executable, "-m", "data.convert", str(tmp), str(out),
         "--min-future-valid", str(min_future_valid)],
        cwd=str(REPO),
        capture_output=True,
    )
    Path(tmp).unlink(missing_ok=True)
    if r.returncode == 0:
        log(f"[{split} {i}/{total}] ok")
    else:
        log(f"[{split} {i}/{total}] CONVERT FAILED: {r.stderr.decode()[-200:]}")
        Path(out).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="+", choices=list(SPLITS))
    ap.add_argument("--shard-root", type=Path,
                    default=Path(os.environ.get("SHARD_ROOT", os.path.expanduser("~/shards"))))
    ap.add_argument("--tmp", type=Path,
                    default=Path(os.environ.get("TFRECORD_TMPDIR", "/tmp/womd_tfrecord")))
    ap.add_argument("--dl-workers", type=int, default=8)
    ap.add_argument("--cv-workers", type=int, default=12)
    ap.add_argument("--min-future-valid", type=int, default=4,
                    help="Drop tracks with fewer than N valid future steps. "
                         "Use 0 for the WOMD test split.")
    args = ap.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)

    jobs = gen_jobs(args.splits, args.shard_root, args.tmp)
    log(f"starting: {len(jobs)} jobs  dl={args.dl_workers} cv={args.cv_workers}  "
        f"root={args.shard_root}")
    if not jobs:
        log("nothing to do.")
        return

    jobs_q: queue.Queue = queue.Queue()
    for j in jobs:
        jobs_q.put(j)
    # Bounded to keep at most (dl + cv_workers + 4) tfrecords on disk.
    ready_q: queue.Queue = queue.Queue(maxsize=args.cv_workers + 4)

    SENTINEL = object()

    def dl_worker() -> None:
        while True:
            j = jobs_q.get()
            if j is SENTINEL:
                jobs_q.task_done()
                break
            try:
                if download(j):
                    ready_q.put(j)
            finally:
                jobs_q.task_done()

    def cv_worker() -> None:
        while True:
            j = ready_q.get()
            if j is SENTINEL:
                ready_q.task_done()
                break
            try:
                convert(j, min_future_valid=args.min_future_valid)
            finally:
                ready_q.task_done()

    dl_threads = [threading.Thread(target=dl_worker, daemon=True, name=f"dl-{k}")
                  for k in range(args.dl_workers)]
    cv_threads = [threading.Thread(target=cv_worker, daemon=True, name=f"cv-{k}")
                  for k in range(args.cv_workers)]
    for t in dl_threads + cv_threads:
        t.start()

    # Phase 1: wait for all downloads to complete.
    # Inject DL sentinels once all real jobs have been taken.
    for _ in dl_threads:
        jobs_q.put(SENTINEL)
    for t in dl_threads:
        t.join()

    # Phase 2: signal CV workers to exit after the ready queue drains.
    for _ in cv_threads:
        ready_q.put(SENTINEL)
    for t in cv_threads:
        t.join()

    log("all done.")


if __name__ == "__main__":
    main()
