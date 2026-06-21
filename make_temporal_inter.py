#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Re)generate the RecBole .inter for ikgr-custom WITH a clipped `timestamp:float`
field, so RecBole can do a temporal (order=TO) split for the DynLLM stage.

Reads data/k_core/interactions_k100_ts.csv (from add_timestamps.py), clips
out-of-range/garbage timestamps to a sane Goodreads window, and writes
data/k_core/ikgr-custom/ikgr-custom.inter (tab-separated atomic format).
RS runs that only load [user_id,item_id,rating] are unaffected (extra column
is ignored unless TIME_FIELD/order=TO is set).
"""
import argparse, csv, os
from datetime import datetime

LO = datetime(2006, 1, 1).timestamp()    # clamp floor (pre-2006 = garbage)
HI = datetime(2017, 12, 31).timestamp()   # clamp ceil (dump era ~2017)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter_ts", default="data/k_core/interactions_k100_ts.csv")
    ap.add_argument("--out", default="data/k_core/ikgr-custom/ikgr-custom.inter")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n, clipped = 0, 0
    with open(args.inter_ts, "r", encoding="utf-8") as fin, \
         open(args.out, "w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        w = csv.writer(fout, delimiter="\t")
        w.writerow(["user_id:token", "item_id:token", "rating:float", "timestamp:float"])
        for row in r:
            n += 1
            try:
                ts = float(row.get("timestamp", 0) or 0)
            except Exception:
                ts = 0.0
            c = min(max(ts, LO), HI) if ts > 0 else LO
            if c != ts:
                clipped += 1
            w.writerow([row["user_id"], row["item_id"], float(row["rating"]), c])
    print(f"[temporal .inter] rows={n} clipped={clipped} -> {args.out}")


if __name__ == "__main__":
    main()
