#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build intent banks from step2 CSV for IKGR.

Inputs:
  --step2_csv   Path to run/step2_related_intents.csv
  --encoder     SentenceTransformers model name (default: all-mpnet-base-v2)
  --user_out    Path to save user bank .pt (default: run/user_bank.pt)
  --item_out    Path to save item bank .pt (default: run/item_bank.pt)

Output format:
  torch.save({
      "encoder": <model name>,
      "dim": <embedding_dim>,
      "bank": { <raw_token: str> : torch.Tensor[k, d] }
  }, path)
"""

import argparse, ast, os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

def _as_list(s):
    if isinstance(s, str) and s.strip():
        try:
            return list(ast.literal_eval(s))
        except Exception:
            try:
                return list(eval(s))
            except Exception:
                return []
    return []

def _unique_keep_order(lst):
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def build_banks(step2_csv, encoder_name, user_out, item_out):
    df = pd.read_csv(step2_csv).fillna("")
    enc = SentenceTransformer(encoder_name)

    user_bank = {}
    item_bank = {}

    for _, r in df.iterrows():
        uid_raw = str(r.get("user_id", "")).strip()
        iid_raw = str(r.get("item_id", "")).strip()
        # collect intents: exact + related
        u_ints = _unique_keep_order(_as_list(r.get("user_intents_exact", "[]")) + _as_list(r.get("user_intents_related", "[]")))
        i_ints = _unique_keep_order(_as_list(r.get("item_intents_exact", "[]")) + _as_list(r.get("item_intents_related", "[]")))

        if uid_raw and u_ints:
            emb_u = enc.encode(u_ints, convert_to_tensor=True)  # [k_u, d]
            user_bank[uid_raw] = emb_u.cpu()
        if iid_raw and i_ints:
            emb_i = enc.encode(i_ints, convert_to_tensor=True)  # [k_i, d]
            item_bank[iid_raw] = emb_i.cpu()

    dim = next(iter(user_bank.values())).shape[1] if user_bank else (next(iter(item_bank.values())).shape[1] if item_bank else 0)

    os.makedirs(os.path.dirname(user_out), exist_ok=True)
    os.makedirs(os.path.dirname(item_out), exist_ok=True)

    torch.save({"encoder": encoder_name, "dim": dim, "bank": user_bank}, user_out)
    torch.save({"encoder": encoder_name, "dim": dim, "bank": item_bank}, item_out)

    print(f"[OK] user_bank → {user_out}  ({len(user_bank)} users, dim={dim})")
    print(f"[OK] item_bank → {item_out}  ({len(item_bank)} items, dim={dim})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2_csv", required=True, help="Path to step2_related_intents.csv")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--user_out", default="run/user_bank.pt")
    ap.add_argument("--item_out", default="run/item_bank.pt")
    args = ap.parse_args()

    build_banks(args.step2_csv, args.encoder, args.user_out, args.item_out)

if __name__ == "__main__":
    main()
