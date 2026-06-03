#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Goodreads JSONL (.json.gz) into pipeline-ready CSV files.

Outputs:
  - profiles.csv: user_id,user_profile,item_id,item_profile
  - interactions.csv: user_id,item_id,rating
"""

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, Tuple


def _clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def _join_parts(parts) -> str:
    return " | ".join([p for p in parts if p])


def _extract_names(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        names = []
        for v in value:
            if isinstance(v, dict) and v.get("name"):
                names.append(str(v["name"]))
            elif isinstance(v, str):
                names.append(v)
        return ", ".join(names)
    return ""


def build_item_profile(book: Dict) -> str:
    title = _clean_text(book.get("title", ""))
    desc = _clean_text(book.get("description", ""))
    authors = _extract_names(book.get("authors", []))
    shelves = _extract_names(book.get("popular_shelves", []))
    return _join_parts([title, authors, desc, shelves])


def load_books(books_gz: str, max_text_len: int) -> Dict[str, str]:
    item_profile: Dict[str, str] = {}
    with gzip.open(books_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            book = json.loads(line)
            book_id = book.get("book_id") or book.get("id")
            if book_id is None:
                continue
            profile = build_item_profile(book)
            if max_text_len > 0:
                profile = profile[:max_text_len]
            item_profile[str(book_id)] = profile
    return item_profile


def iter_interactions(inter_gz: str) -> Iterable[Tuple[str, str, float]]:
    with gzip.open(inter_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("user_id")
            iid = row.get("book_id") or row.get("item_id")
            rating = row.get("rating", 0)
            if uid is None or iid is None:
                continue
            try:
                rating = float(rating)
            except Exception:
                rating = 0.0
            yield str(uid), str(iid), rating


def write_interactions(inter_gz: str, out_interactions: str, min_rating: float = None) -> None:
    os.makedirs(os.path.dirname(out_interactions), exist_ok=True)
    with open(out_interactions, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "rating"])
        for uid, iid, rating in iter_interactions(inter_gz):
            if min_rating is not None and rating < min_rating:
                continue
            writer.writerow([uid, iid, rating])


def build_user_profiles(inter_gz: str,
                        item_profile: Dict[str, str],
                        max_profile_items: int,
                        max_text_len: int,
                        min_rating: float = None) -> Dict[str, str]:
    user_items: Dict[str, list] = defaultdict(list)

    for uid, iid, rating in iter_interactions(inter_gz):
        if min_rating is not None and rating < min_rating:
            continue
        if len(user_items[uid]) >= max_profile_items:
            continue
        if iid in item_profile:
            user_items[uid].append(iid)

    user_profile: Dict[str, str] = {}
    for uid, iids in user_items.items():
        texts = [item_profile.get(iid, "") for iid in iids if item_profile.get(iid, "")]
        profile = " ; ".join(texts)
        if max_text_len > 0:
            profile = profile[:max_text_len]
        user_profile[uid] = profile

    return user_profile


def write_profiles(inter_gz: str,
                   out_profiles: str,
                   user_profile: Dict[str, str],
                   item_profile: Dict[str, str],
                   min_rating: float = None) -> None:
    os.makedirs(os.path.dirname(out_profiles), exist_ok=True)
    with open(out_profiles, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "user_profile", "item_id", "item_profile"])
        for uid, iid, rating in iter_interactions(inter_gz):
            if min_rating is not None and rating < min_rating:
                continue
            writer.writerow([
                uid,
                user_profile.get(uid, ""),
                iid,
                item_profile.get(iid, "")
            ])


def main():
    ap = argparse.ArgumentParser(description="Convert Goodreads JSONL (.json.gz) to IKGR pipeline CSVs")
    ap.add_argument("--interactions_gz", required=True, help="Path to goodreads_interactions_*.json.gz")
    ap.add_argument("--books_gz", required=True, help="Path to goodreads_books_*.json.gz")
    ap.add_argument("--out_profiles", default="data/profiles.csv")
    ap.add_argument("--out_interactions", default="data/interactions.csv")
    ap.add_argument("--max_profile_items", type=int, default=20, help="Max items per user used to build user_profile")
    ap.add_argument("--max_text_len", type=int, default=2000, help="Max length for item/user profile text")
    ap.add_argument("--min_rating", type=float, default=None, help="Optional: filter interactions by minimum rating")
    args = ap.parse_args()

    print("[1/3] Loading books...")
    item_profile = load_books(args.books_gz, args.max_text_len)
    print(f"[books] {len(item_profile)} items")

    print("[2/3] Writing interactions.csv...")
    write_interactions(args.interactions_gz, args.out_interactions, min_rating=args.min_rating)
    print(f"[interactions] saved -> {args.out_interactions}")

    print("[3/3] Building user profiles + writing profiles.csv...")
    user_profile = build_user_profiles(
        args.interactions_gz,
        item_profile,
        args.max_profile_items,
        args.max_text_len,
        min_rating=args.min_rating,
    )
    write_profiles(args.interactions_gz, args.out_profiles, user_profile, item_profile, min_rating=args.min_rating)
    print(f"[profiles] saved -> {args.out_profiles}")


if __name__ == "__main__":
    main()
