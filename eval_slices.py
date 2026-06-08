#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliced evaluation: does the intent KG help on COLD-START users / LONG-TAIL items?

For each model (IKGR KG-on, IKGR KG-off, BPR, LightGCN), trained on the SAME
split/seed/protocol as step3, compute per-user ranking metrics and slice by:
  - user activity   = #history items masked at eval (train+valid degree) -> cold-start
  - item popularity = #times an item is a history item across users       -> long-tail

We store, per test user, the top-K predicted item ids + the relevant (test) item
set, then compute overall / cold-start-bucket / long-tail metrics in one place.

Output: run/slice_eval_result.json    Run: python eval_slices.py
"""
import os, json, time, math, yaml
# compat: RecBole 1.2.0 calls scipy dok_matrix._update(), removed in newer scipy
import scipy.sparse as _sp
if not hasattr(_sp.dok_matrix, "_update"):
    _sp.dok_matrix._update = dict.update
import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer
from ikgr_core.model_ikgr import IKGR as IKGRModel

KS = [10, 30]
MAXK = max(KS)


def _config(rb, paths, extra):
    cd = {
        "epochs": rb["epochs"], "metrics": rb["metrics"], "topk": rb["topk"],
        "embedding_size": rb["embedding_size"], "learning_rate": 1e-3, "reg_weight": 1e-6,
        "dropout_prob": rb.get("dropout", 0.1),
        "data_path": os.path.dirname(paths["inter_file"]),
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "LABEL_FIELD": "rating",
        "load_col": {"inter": ["user_id", "item_id", "rating"]},
        "train_neg_sample_args": {"distribution": "uniform"},
        "save_dataset": False, "save_dataloaders": False, "show_progress": False,
        "checkpoint_dir": os.path.abspath("run/recbole_slice"), "eval_step": 5,
        "seed": 2020, "reproducibility": True,
    }
    cd.update(extra)
    return cd


def _train_and_collect(model_arg, extra, rb, paths):
    """Train a model; return per-user (uid, activity, topk items, relevant items)
    plus item popularity (from masked history)."""
    os.makedirs("run/recbole_slice", exist_ok=True)
    config = Config(model=model_arg, dataset=rb["dataset"], config_dict=_config(rb, paths, extra))
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    klass = model_arg if not isinstance(model_arg, str) else get_model(config["model"])
    model = klass(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    t0 = time.time()
    trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    train_sec = round(time.time() - t0, 1)
    smf = getattr(trainer, "saved_model_file", None)
    if smf and os.path.exists(smf):
        ck = torch.load(smf, map_location=config["device"])
        model.load_state_dict(ck["state_dict"])
        if ck.get("other_parameter"):
            model.load_other_parameter(ck["other_parameter"])
    model.eval()

    dev = config["device"]
    n_items = int(dataset.item_num)
    uid_field = config["USER_ID_FIELD"]
    item_pop = np.zeros(n_items, dtype=np.int64)
    per_user = []  # dict(uid, activity, topk(list[int]), rel(list[int]))
    with torch.no_grad():
        for interaction, history_index, positive_u, positive_i in test_data:
            interaction = interaction.to(dev)
            users = interaction[uid_field]
            B = users.shape[0]
            scores = model.full_sort_predict(interaction).view(B, -1)  # [B, n_items]
            scores[:, 0] = -np.inf
            if history_index is not None:
                hr = history_index[0].cpu().numpy(); hc = history_index[1].cpu().numpy()
                np.add.at(item_pop, hc, 1)
                ucnt = np.bincount(hr, minlength=B)
                scores[history_index] = -np.inf
            else:
                ucnt = np.zeros(B, dtype=np.int64)
            topk = torch.topk(scores, MAXK, dim=-1)[1].cpu().numpy()
            users_np = users.cpu().numpy()
            pu = positive_u.cpu().numpy(); pi = positive_i.cpu().numpy()
            rel_by_row = {}
            for r, it in zip(pu, pi):
                rel_by_row.setdefault(int(r), []).append(int(it))
            for row in range(B):
                rel = rel_by_row.get(row)
                if not rel:
                    continue
                per_user.append({"uid": int(users_np[row]), "activity": int(ucnt[row]),
                                 "topk": topk[row].tolist(), "rel": rel})
    return per_user, item_pop, train_sec, n_items


def _user_metrics(topk, rel_set, k):
    hits = [rank for rank, it in enumerate(topk[:k]) if it in rel_set]
    n_rel = len(rel_set)
    recall = len(hits) / n_rel if n_rel else 0.0
    hit = 1.0 if hits else 0.0
    dcg = sum(1.0 / math.log2(r + 2) for r in hits)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg, hit


def _avg_metrics(rows, is_tail=None):
    """rows: list of (topk, rel_list). If is_tail given, restrict rel to tail items."""
    agg = {f"recall@{k}": 0.0 for k in KS}
    agg.update({f"ndcg@{k}": 0.0 for k in KS}); agg.update({f"hit@{k}": 0.0 for k in KS})
    n = 0
    for topk, rel in rows:
        if is_tail is not None:
            rel = [it for it in rel if is_tail[it]]
            if not rel:
                continue
        rel_set = set(rel)
        n += 1
        for k in KS:
            rc, nd, ht = _user_metrics(topk, rel_set, k)
            agg[f"recall@{k}"] += rc; agg[f"ndcg@{k}"] += nd; agg[f"hit@{k}"] += ht
    if n:
        for key in agg:
            agg[key] = round(agg[key] / n, 4)
    return agg, n


def slice_report(per_user, item_pop):
    # long-tail items = not in top 20% by popularity (head_cut = 80th pct of >0 pops)
    pos = item_pop[item_pop > 0]
    head_cut = float(np.percentile(pos, 80)) if pos.size else 0.0
    is_tail = item_pop <= head_cut

    rows_all = [(r["topk"], r["rel"]) for r in per_user]
    overall, n_all = _avg_metrics(rows_all)

    # cold-start buckets by user activity quantiles
    acts = np.array([r["activity"] for r in per_user])
    q = np.percentile(acts, [20, 40, 60, 80]) if acts.size else [0, 0, 0, 0]
    def bucket(a):
        return ("Q1_cold" if a <= q[0] else "Q2" if a <= q[1] else "Q3" if a <= q[2]
                else "Q4" if a <= q[3] else "Q5_warm")
    buckets = {}
    for r in per_user:
        buckets.setdefault(bucket(r["activity"]), []).append((r["topk"], r["rel"]))
    bucket_metrics = {}
    for b, rows in buckets.items():
        m, n = _avg_metrics(rows)
        bucket_metrics[b] = {"n_users": n, **{kk: m[kk] for kk in ("recall@10", "ndcg@10", "hit@10")}}

    tail_metrics, n_tail = _avg_metrics(rows_all, is_tail=is_tail)
    return {
        "overall": overall,
        "n_test_users": n_all,
        "cold_start_buckets": dict(sorted(bucket_metrics.items())),
        "activity_quantiles_20_40_60_80": [float(x) for x in q],
        "long_tail": {"head_cut_pop": head_cut, "n_tail_items": int(is_tail.sum()),
                      "n_users_with_tail_rel": n_tail, **tail_metrics},
    }


def main():
    cfg = yaml.safe_load(open("config.yaml")); paths, rb = cfg["paths"], cfg["recbole"]
    kg_pack = os.path.abspath("run/kg_pack.pt")
    specs = [
        ("IKGR_kgon", IKGRModel, {"use_kg": True, "kg_pack_path": kg_pack}),
        ("IKGR_kgoff", IKGRModel, {"use_kg": False}),
        ("BPR", "BPR", {}),
        ("LightGCN", "LightGCN", {}),
    ]
    out_path = "run/slice_eval_result.json"
    results = {}
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path, encoding="utf-8"))
        except Exception:
            results = {}

    for name, model_arg, extra in specs:
        if name in results:
            print(f"[skip] {name}", flush=True); continue
        print(f"\n===== {name} =====", flush=True)
        per_user, item_pop, train_sec, n_items = _train_and_collect(model_arg, extra, rb, paths)
        rep = slice_report(per_user, item_pop)
        rep["train_sec"] = train_sec; rep["n_items"] = n_items
        results[name] = rep
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[{name}] overall={rep['overall']['ndcg@10']} "
              f"cold(Q1)={rep['cold_start_buckets'].get('Q1_cold',{}).get('ndcg@10')} "
              f"tail_recall@10={rep['long_tail']['recall@10']} ({train_sec}s)", flush=True)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
