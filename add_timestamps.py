#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill timestamps into an existing k-core interactions file WITHOUT re-running
the full preprocess + k-core (keeps the exact same rows).

Streams goodreads_interactions_children.json.gz and, for each (user_id, book_id)
present in the k-core file, records an epoch timestamp (prefer read_at, else
date_added). Writes <out> with an added `timestamp` column.

Usage:
  python add_timestamps.py --inter data/k_core/interactions_k100.csv \
      --raw goodreads_interactions_children.json.gz \
      --out data/k_core/interactions_k100_ts.csv
"""
import argparse, csv, gzip, json
from datetime import datetime


def parse_ts(s):
    if not s:
        return 0.0
    try:
        return datetime.strptime(str(s).strip(), "%a %b %d %H:%M:%S %z %Y").timestamp()
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", default="data/k_core/interactions_k100.csv")
    ap.add_argument("--raw", default="goodreads_interactions_children.json.gz")
    ap.add_argument("--out", default="data/k_core/interactions_k100_ts.csv")
    args = ap.parse_args()

    # 1) collect needed (user,item) pairs from the k-core file
    needed = set()
    rows = []
    with open(args.inter, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        for row in r:
            key = row["user_id"] + "\t" + row["item_id"]
            needed.add(key)
            rows.append(row)
    print(f"[ts] k-core rows={len(rows)} unique (user,item) pairs={len(needed)}", flush=True)

    # 2) stream raw interactions, fill timestamps for needed pairs
    ts_map = {}
    scanned = 0
    with gzip.open(args.raw, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            if scanned % 1_000_000 == 0:
                print(f"  scanned {scanned:,} raw | filled {len(ts_map):,}/{len(needed):,}", flush=True)
            row = json.loads(line)
            uid = row.get("user_id"); iid = row.get("book_id") or row.get("item_id")
            if uid is None or iid is None:
                continue
            key = str(uid) + "\t" + str(iid)
            if key not in needed or key in ts_map:
                continue
            ts_map[key] = parse_ts(row.get("read_at") or row.get("date_added") or "")
            if len(ts_map) >= len(needed):
                print("  all pairs filled, stopping early.", flush=True)
                break

    filled = sum(1 for k in needed if k in ts_map)
    nonzero = sum(1 for v in ts_map.values() if v > 0)
    print(f"[ts] filled {filled}/{len(needed)} pairs ({nonzero} non-zero ts)", flush=True)

    # 3) write out with timestamp column
    out_cols = list(cols) + ["timestamp"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for row in rows:
            key = row["user_id"] + "\t" + row["item_id"]
            row["timestamp"] = ts_map.get(key, 0.0)
            w.writerow(row)
    print(f"[ts] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
