#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the HETEROGENEOUS metadata KG pack for IKGR (diagram nodes:
Brand / Category-Attribute), with NO LLM calls -- purely from book metadata.

For every item in our k=100 dataset (item tokens taken from run/kg_pack.pt), pull
from goodreads_books_children.json.gz:
  - authors  (author_id)  -> "brand" nodes, relation item--by_author
  - publisher             -> "brand" nodes, relation item--by_publisher
  - popular_shelves       -> "shelf" nodes (category/attribute), item--has_shelf
    (reading-status / non-content shelves filtered out; top-K by count kept)

Output run/meta_kg_pack.pt = {
  "item_authors":    {item_token(str): LongTensor[author node ids]},
  "item_publishers": {item_token(str): LongTensor[publisher node ids]},
  "item_shelves":    {item_token(str): LongTensor[shelf node ids]},
  "n_authors": int, "n_publishers": int, "n_shelves": int,
  "author_vocab": [...], "publisher_vocab": [...], "shelf_vocab": [...],
}
Node ids index per-relation vocabularies (each relation has its own id space).
"""
import argparse, gzip, json, os
from collections import defaultdict, Counter
import torch

# Reading-status / non-content shelves to drop (not genres/attributes).
SHELF_STOP = {
    "to-read", "currently-reading", "read", "owned", "owned-books", "books-i-own",
    "favorites", "favourites", "default", "to-buy", "wish-list", "wishlist",
    "my-books", "my-library", "library", "re-read", "reread", "dnf", "abandoned",
    "have-read", "finished", "unfinished", "tbr", "to-read-owned", "kindle",
    "ebook", "ebooks", "audiobook", "audiobooks", "audio", "audible", "owned-tbr",
    "i-own", "want-to-read", "books", "book", "all-books", "general", "unread",
    "to-read-fiction", "maybe", "to-read-maybe", "not-read", "did-not-finish",
}


def _top_shelves(shelves, k, min_count):
    out = []
    for s in shelves or []:
        name = (s.get("name") or "").strip().lower()
        try:
            cnt = int(s.get("count", 0))
        except Exception:
            cnt = 0
        if not name or name in SHELF_STOP or cnt < min_count:
            continue
        out.append((cnt, name))
    out.sort(reverse=True)
    return [name for _, name in out[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books_gz", default="goodreads_books_children.json.gz")
    ap.add_argument("--kg_pack", default="run/kg_pack.pt", help="source of item token set")
    ap.add_argument("--out", default="run/meta_kg_pack.pt")
    ap.add_argument("--top_shelves", type=int, default=15)
    ap.add_argument("--min_shelf_count", type=int, default=3)
    ap.add_argument("--min_shelf_df", type=int, default=5, help="drop shelves appearing in < df items")
    args = ap.parse_args()

    item_set = set(torch.load(args.kg_pack, map_location="cpu")["item_intents"].keys())
    item_set = {str(x) for x in item_set}
    print(f"[meta] target items: {len(item_set)}")

    raw = {}  # item -> (authors[list], publisher, shelves[list])
    shelf_df = Counter()
    n_seen = 0
    with gzip.open(args.books_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            bid = str(b.get("book_id") or b.get("id") or "")
            if bid not in item_set:
                continue
            n_seen += 1
            authors = [str(a.get("author_id")) for a in (b.get("authors") or []) if a.get("author_id")]
            pub = (b.get("publisher") or "").strip()
            shelves = _top_shelves(b.get("popular_shelves"), args.top_shelves, args.min_shelf_count)
            for s in shelves:
                shelf_df[s] += 1
            raw[bid] = (authors, pub, shelves)
    print(f"[meta] matched {len(raw)}/{len(item_set)} items in books file (seen {n_seen})")

    # drop rare shelves (document frequency too low)
    keep_shelf = {s for s, df in shelf_df.items() if df >= args.min_shelf_df}
    print(f"[meta] shelves: {len(shelf_df)} unique -> {len(keep_shelf)} kept (df>={args.min_shelf_df})")

    author_vocab, pub_vocab, shelf_vocab = {}, {}, {}
    def vid(vocab, key):
        if key not in vocab:
            vocab[key] = len(vocab)
        return vocab[key]

    item_authors, item_publishers, item_shelves = {}, {}, {}
    for bid, (authors, pub, shelves) in raw.items():
        a_ids = [vid(author_vocab, a) for a in authors]
        p_ids = [vid(pub_vocab, pub)] if pub else []
        s_ids = [vid(shelf_vocab, s) for s in shelves if s in keep_shelf]
        if a_ids:
            item_authors[bid] = torch.tensor(a_ids, dtype=torch.long)
        if p_ids:
            item_publishers[bid] = torch.tensor(p_ids, dtype=torch.long)
        if s_ids:
            item_shelves[bid] = torch.tensor(s_ids, dtype=torch.long)

    inv = lambda v: [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
    pack = {
        "item_authors": item_authors, "item_publishers": item_publishers, "item_shelves": item_shelves,
        "n_authors": len(author_vocab), "n_publishers": len(pub_vocab), "n_shelves": len(shelf_vocab),
        "author_vocab": inv(author_vocab), "publisher_vocab": inv(pub_vocab), "shelf_vocab": inv(shelf_vocab),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(pack, args.out)
    ea = sum(len(v) for v in item_authors.values())
    ep = sum(len(v) for v in item_publishers.values())
    es = sum(len(v) for v in item_shelves.values())
    print(f"[meta] authors={len(author_vocab)} (item-edges {ea}) | publishers={len(pub_vocab)} (edges {ep}) "
          f"| shelves={len(shelf_vocab)} (edges {es})")
    print(f"[meta] items with author={len(item_authors)} pub={len(item_publishers)} shelf={len(item_shelves)}")
    print(f"[meta] saved -> {args.out}")


if __name__ == "__main__":
    main()
