#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert standard CSV files into RecBole atomic dataset format:
- interactions.csv  -->  dataset_name.inter
- step2_related_intents.csv  -->  dataset_name.kg

Usage:
    python convert_to_recbole_atomic.py \
        --interactions data/interactions.csv \
        --intents run/step2_related_intents.csv \
        --out_dir data \
        --dataset ikgr-custom
"""

import os
import csv
import argparse
import pandas as pd

def _safe_eval_list(s):
    """Safely parse a stringified Python list like '["a","b"]'."""
    if isinstance(s, str) and s.strip():
        try:
            return list(eval(s))
        except Exception:
            return []
    return []

def to_recbole_inter(interactions_csv: str, out_dir: str, dataset_name: str):
    """Convert interactions.csv to RecBole .inter format."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}.inter")
    df = pd.read_csv(interactions_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["user_id:token", "item_id:token", "rating:float"])
        for _, row in df.iterrows():
            writer.writerow([
                str(row["user_id"]),
                str(row["item_id"]),
                float(row["rating"])
            ])
    print(f"[OK] .inter file saved → {out_path}")

def to_recbole_kg(intent_csv: str, out_dir: str, dataset_name: str):
    """Convert step2_related_intents.csv to RecBole .kg format."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}.kg")
    df = pd.read_csv(intent_csv).fillna("")
    cols_user = ["user_intents_exact", "user_intents_related"]
    cols_item = ["item_intents_exact", "item_intents_related"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["head_id:token", "relation_id:token", "tail_id:token"])
        for _, row in df.iterrows():
            uid = f"u_{row['user_id']}" if "user_id" in row else None
            iid = f"i_{row['item_id']}" if "item_id" in row else None

            # User → intent
            if uid:
                intents = []
                for c in cols_user:
                    intents.extend(_safe_eval_list(row.get(c, "[]")))
                for it in sorted(set(intents)):
                    writer.writerow([uid, "user_has_intent", f"intent::{it}"])

            # Item → intent
            if iid:
                intents = []
                for c in cols_item:
                    intents.extend(_safe_eval_list(row.get(c, "[]")))
                for it in sorted(set(intents)):
                    writer.writerow([iid, "item_has_intent", f"intent::{it}"])
    print(f"[OK] .kg file saved → {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to RecBole atomic format (.inter, .kg)")
    parser.add_argument("--interactions", required=True, help="Path to interactions.csv (user_id,item_id,rating)")
    parser.add_argument("--intents", required=True, help="Path to step2_related_intents.csv (with intent columns)")
    parser.add_argument("--out_dir", required=True, help="Output directory for .inter and .kg")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., ikgr-custom)")
    args = parser.parse_args()

    to_recbole_inter(args.interactions, args.out_dir, args.dataset)
    to_recbole_kg(args.intents, args.out_dir, args.dataset)

if __name__ == "__main__":
    main()
