"""Publish immutable reports after registered remote batches terminate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

from collect_uq_expansion_20261013 import collect
from summarize_uq_expansion_20261013 import export_csv, make_figures, summarize, write_report
from verify_uq_expansion_independent_20261013 import verify
from write_uq_expansion_report_zh_20261013 import write_report as write_chinese_report


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def terminal_batches(root):
    names = {
        "expansion": "results/uq_baseline_expansion_20261013/native.status",
        "tag_references": "results/uq_final_evidence_20261012/native_parallel_references.status",
        "gecc_soft": "results/uq_final_evidence_20261012/native_parallel_gecc.status",
    }
    terminal = {}
    for name, relative in names.items():
        path = root / relative
        if path.exists():
            content = path.read_text().strip()
            if content.startswith(("complete ", "references_complete ", "failed ", "blocked_")):
                terminal[name] = content
    return terminal


def publish(root, markers):
    destination = root / "results/uq_baseline_expansion_20261013"
    if shutil.disk_usage(destination).free < 2 * 1024**3:
        raise RuntimeError("Reporting paused: less than 2 GiB available; no files deleted")
    stamp = timestamp()
    snapshot_path = destination / f"expansion_utc_{stamp}.json"
    report_path = destination / f"report_expansion_utc_{stamp}"
    snapshot = collect(root)
    bundle = summarize(snapshot)
    independent = verify(snapshot, bundle)
    report_path.mkdir()
    atomic_json(snapshot_path, snapshot)
    for key in ("matched", "stability", "paired", "coverage", "stages", "sources", "native_diagnostics"):
        export_csv(report_path / f"{key}.csv", [
            {k: v for k, v in row.items() if k != "per_student"} for row in bundle[key]])
    atomic_json(report_path / "bundle.json", bundle)
    atomic_json(report_path / "verification.json", bundle["checks"])
    atomic_json(report_path / "independent_verification.json", independent)
    write_report(bundle, report_path)
    write_chinese_report(bundle, report_path)
    make_figures(bundle, report_path)
    completed = sum(row["status"] == "complete" for row in bundle["coverage"])
    receipt = dict(snapshot=str(snapshot_path.relative_to(root)),
        report=str(report_path.relative_to(root)), collected_at=bundle["collected_at"],
        completed=completed, registered=len(bundle["coverage"]), markers=markers,
        all_registered_complete=completed == len(bundle["coverage"]), checks=bundle["checks"],
        independent_verification=independent["status"])
    atomic_json(destination / "latest_report_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=72)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 60 or args.max_hours <= 0:
        parser.error("poll-seconds must be 10..60 and max-hours must be positive")
    directory = args.root / "results/uq_baseline_expansion_20261013"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "report_watcher_state.json"
    with (directory / "report_watcher.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An existing report watcher owns this queue")
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        seen = previous.get("published_markers", {})
        started = time.monotonic()
        while True:
            markers = terminal_batches(args.root)
            state = dict(status="watching", updated_at=datetime.now(timezone.utc).isoformat(),
                         published_markers=seen, observed_markers=markers)
            try:
                if args.once or markers != seen:
                    receipt = publish(args.root, markers)
                    seen = markers
                    state.update(published_markers=seen, latest_report=receipt)
                    if receipt["all_registered_complete"]:
                        state["status"] = "complete"
                    elif len(markers) == 3:
                        state["status"] = "batches_terminated_with_missing_results"
                atomic_json(state_path, state)
            except Exception as error:
                state.update(status="report_failed", error=f"{type(error).__name__}: {error}")
                atomic_json(state_path, state)
                print(json.dumps(state), flush=True)
                raise
            if args.once or state["status"] in ("complete", "batches_terminated_with_missing_results"):
                return
            if time.monotonic() - started >= args.max_hours * 3600:
                state["status"] = "timed_out_with_pending_results"
                atomic_json(state_path, state)
                raise SystemExit(1)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
