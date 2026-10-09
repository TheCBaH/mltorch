#!/usr/bin/env python3
"""Aggregate matrix over the cohort: one row per artifact and backend.

Usage: transformers-matrix.py COHORT_JSON REPORT_DIR...

Offline. Reads the cohort manifest and the replay reports (*.replay.json) and
prints a Markdown table. The graph-only admission sweep is over the producer
catalogue's own population ("tiny"), not the release's ("reference"), so its
rows are not joined here. A cell is exactly
one of: passed, failed (numeric, with counts), refused (graph or dialect), or
not run. Nothing is inferred: a missing report is "not run".
"""
import glob
import json
import os
import sys


def main(cohort_path, *report_dirs):
    cohort = json.load(open(cohort_path))
    reports = {}
    for d in report_dirs:
        for f in glob.glob(os.path.join(d, "*.replay.json")):
            r = json.load(open(f))
            reports.setdefault(r["artifact_id"], {})[r["backend"]] = r
    print("| artifact | backend | status | detail |")
    print("|---|---|---|---|")
    for a in cohort["artifacts"]:
        aid = a["artifact_id"]
        short = aid.split("/ckpt")[0]
        by_backend = reports.get(aid)
        if not by_backend:
            print(f"| {short} | - | not run | |")
            continue
        for backend, r in sorted(by_backend.items()):
            detail = r.get("refusal") or ""
            if r["status"] == "failed":
                bad = [
                    f'{c["id"]} {o["name"]}: {o["mismatches"]}/{o["elements"]}'
                    for c in r["cases"]
                    for o in c["outputs"]
                    if o["verdict"] != "pass"
                ]
                detail = "; ".join(bad[:3])
            print(f"| {short} | {backend} | {r['status']} | {detail} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
