#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed sliced evaluation for the IKGR "squeeze": overall vs long-tail vs
coverage/novelty, with robustness across seeds.

Models: IKGR KG-off (MF), IKGR KG-on L1, IKGR KG-on L2, BPR, LightGCN.
For each model x seed: train on the same protocol, compute per-user metrics,
slice by user activity (cold-start) and item popularity (long-tail), plus
catalog coverage@10 and recommended-item novelty. Aggregate mean +/- std over
seeds.  Output: run/slice_eval_result.json

Env overrides (for smoke tests):
  IKGR_SEEDS=2020            comma-separated seeds (default 2020,2021,2022)
  IKGR_EPOCHS=2              epochs override
  IKGR_SPECS=IKGR_kgon_L2,IKGR_kgoff   subset of specs
"""
import os, json, time, math, yaml
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
COV_K = 10


def _config(rb, paths, extra, seed):
    split = os.environ.get("IKGR_SPLIT", "RS").upper()
    load_inter = ["user_id", "item_id", "rating"] + (["timestamp"] if split == "TO" else [])
    cd = {
        "epochs": int(os.environ.get("IKGR_EPOCHS", rb["epochs"])),
        "metrics": rb["metrics"], "topk": rb["topk"],
        "embedding_size": rb["embedding_size"], "learning_rate": 1e-3, "reg_weight": 1e-6,
        "dropout_prob": rb.get("dropout", 0.1),
        "data_path": os.path.dirname(paths["inter_file"]),
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "LABEL_FIELD": "rating",
        "load_col": {"inter": load_inter},
        "train_neg_sample_args": {"distribution": "uniform"},
        "save_dataset": False, "save_dataloaders": False, "show_progress": False,
        "checkpoint_dir": os.path.abspath("run/recbole_slice"), "eval_step": 5,
        "seed": seed, "reproducibility": True,
    }
    if split == "TO":
        # temporal (per-user time-ordered) split: train=earliest, test=latest
        cd["TIME_FIELD"] = "timestamp"
        cd["eval_args"] = {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO",
                           "group_by": "user", "mode": "full"}
    cd.update(extra)
    return cd


def _build_recency(model, train_data, config):
    """Per-user recency profile from TRAIN interactions only (no leakage):
    top-N most recent items + exp-decay weights -> model.set_recency()."""
    from collections import defaultdict
    inter = train_data.dataset.inter_feat
    uf, itf = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]
    if "timestamp" not in inter.columns:
        raise RuntimeError("use_dynamic requires timestamp (run with IKGR_SPLIT=TO).")
    u = inter[uf].numpy(); it = inter[itf].numpy(); ts = inter["timestamp"].numpy()
    N = model.recency_topn
    tau = max(model.recency_tau_days * 86400.0, 1.0)
    byu = defaultdict(list)
    for a, b, c in zip(u, it, ts):
        byu[int(a)].append((float(c), int(b)))
    ids = np.zeros((model.n_users, N), dtype=np.int64)
    wts = np.zeros((model.n_users, N), dtype=np.float32)
    for uid, lst in byu.items():
        lst.sort(reverse=True)            # most recent first
        lst = lst[:N]
        tref = lst[0][0]
        w = np.exp(-np.array([tref - c for c, _ in lst], dtype=np.float64) / tau)
        s = w.sum()
        w = (w / s) if s > 0 else w
        for j, (c, b) in enumerate(lst):
            ids[uid, j] = b; wts[uid, j] = w[j]
    dev = next(model.parameters()).device
    model.set_recency(torch.from_numpy(ids).to(dev), torch.from_numpy(wts).to(dev))
    print(f"  [recency] built for {len(byu)} users from {len(u)} train inters (N={N})", flush=True)


def _train_and_collect(model_arg, extra, rb, paths, seed):
    os.makedirs("run/recbole_slice", exist_ok=True)
    config = Config(model=model_arg, dataset=rb["dataset"], config_dict=_config(rb, paths, extra, seed))
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    klass = model_arg if not isinstance(model_arg, str) else get_model(config["model"])
    model = klass(config, train_data.dataset).to(config["device"])
    if getattr(model, "use_dynamic", False):
        _build_recency(model, train_data, config)
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
    per_user = []
    with torch.no_grad():
        for interaction, history_index, positive_u, positive_i in test_data:
            interaction = interaction.to(dev)
            users = interaction[uid_field]
            B = users.shape[0]
            scores = model.full_sort_predict(interaction).view(B, -1)
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
    hits = [r for r, it in enumerate(topk[:k]) if it in rel_set]
    n_rel = len(rel_set)
    recall = len(hits) / n_rel if n_rel else 0.0
    dcg = sum(1.0 / math.log2(r + 2) for r in hits)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg, (1.0 if hits else 0.0)


def _avg(rows, is_tail=None):
    agg = {}
    for k in KS:
        agg[f"recall@{k}"] = 0.0; agg[f"ndcg@{k}"] = 0.0; agg[f"hit@{k}"] = 0.0
    n = 0
    for topk, rel in rows:
        if is_tail is not None:
            rel = [it for it in rel if is_tail[it]]
            if not rel:
                continue
        rel_set = set(rel); n += 1
        for k in KS:
            rc, nd, ht = _user_metrics(topk, rel_set, k)
            agg[f"recall@{k}"] += rc; agg[f"ndcg@{k}"] += nd; agg[f"hit@{k}"] += ht
    if n:
        for kk in agg:
            agg[kk] = round(agg[kk] / n, 4)
    return agg, n


def slice_report(per_user, item_pop, n_items):
    pos = item_pop[item_pop > 0]
    head_cut = float(np.percentile(pos, 80)) if pos.size else 0.0
    is_tail = item_pop <= head_cut

    rows_all = [(r["topk"], r["rel"]) for r in per_user]
    overall, n_all = _avg(rows_all)
    tail, n_tail = _avg(rows_all, is_tail=is_tail)

    # cold-start buckets by activity quantiles
    acts = np.array([r["activity"] for r in per_user])
    q = np.percentile(acts, [20, 40, 60, 80]) if acts.size else [0, 0, 0, 0]
    def b(a):
        return ("Q1_cold" if a <= q[0] else "Q2" if a <= q[1] else "Q3" if a <= q[2]
                else "Q4" if a <= q[3] else "Q5_warm")
    buckets = {}
    for r in per_user:
        buckets.setdefault(b(r["activity"]), []).append((r["topk"], r["rel"]))
    bucket_metrics = {}
    for name, rows in buckets.items():
        m, n = _avg(rows)
        bucket_metrics[name] = {"n_users": n, "ndcg@10": m["ndcg@10"], "recall@10": m["recall@10"]}

    # coverage@10 and novelty (mean self-information of recommended items)
    rec_items = set()
    pop_sum, pop_cnt = 0.0, 0
    total = max(1, int(item_pop.sum()))
    novelty = 0.0
    for r in per_user:
        for it in r["topk"][:COV_K]:
            rec_items.add(it)
            p = item_pop[it] / total
            novelty += -math.log2(p) if p > 0 else 0.0
            pop_sum += item_pop[it]; pop_cnt += 1
    coverage = round(len(rec_items) / max(1, n_items - 1), 4)  # exclude pad item
    novelty = round(novelty / max(1, pop_cnt), 4)
    avg_rec_pop = round(pop_sum / max(1, pop_cnt), 1)

    return {
        "overall": overall, "n_test_users": n_all,
        "long_tail": {"head_cut_pop": head_cut, "n_tail_items": int(is_tail.sum()),
                      "n_users_with_tail_rel": n_tail, **tail},
        "cold_start_buckets": dict(sorted(bucket_metrics.items())),
        "activity_q_20_40_60_80": [float(x) for x in q],
        "coverage@10": coverage, "novelty_bits": novelty, "avg_rec_popularity": avg_rec_pop,
    }


def _aggregate(per_seed):
    """mean/std over seeds for headline metrics."""
    keys = [("overall", "ndcg@10"), ("overall", "recall@10"),
            ("long_tail", "recall@10"), ("long_tail", "recall@30"),
            ("coverage@10", None), ("novelty_bits", None)]
    out = {}
    for grp, sub in keys:
        vals = []
        for rep in per_seed.values():
            v = rep[grp] if sub is None else rep[grp][sub]
            vals.append(float(v))
        label = grp if sub is None else f"{grp}.{sub}"
        out[label] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                      "seeds": vals}
    return out


def main():
    cfg = yaml.safe_load(open("config.yaml")); paths, rb = cfg["paths"], cfg["recbole"]
    kg = os.path.abspath("run/kg_pack.pt")
    meta = os.path.abspath("run/meta_kg_pack.pt")
    all_specs = {
        "IKGR_kgoff":   (IKGRModel, {"use_kg": False}),
        "IKGR_kgon_L1": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32}),
        "IKGR_kgon_L1_frozen": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1,
                                            "kg_cap": 32, "intent_learnable": False}),
        "IKGR_kgon_L2": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 2, "kg_cap": 32}),
        # heterogeneous metadata KG (brand/category/attribute), LLM-free
        "IKGR_meta_only": (IKGRModel, {"use_kg": False, "use_meta_kg": True, "meta_kg_path": meta, "kg_cap": 32}),
        "IKGR_full_hetero": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                         "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta}),
        "IKGR_dyn": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                 "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                 "use_dynamic": True}),
        "IKGR_dyn_attn": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                      "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                      "use_dynamic": True, "profile_attn": True}),
        # CORONA stage 3 = Full(IKGR+DynLLM+CORONA): per-channel weighted-sum
        # late-fusion (CF / intent+meta-KG / recency) with learnable alpha,beta,gamma.
        "IKGR_full": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                  "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                  "use_dynamic": True, "use_corona": True}),
        "BPR":          ("BPR", {}),
        "LightGCN":     ("LightGCN", {}),
    }
    seeds = [int(s) for s in os.environ.get("IKGR_SEEDS", "2020,2021,2022").split(",")]
    spec_names = os.environ.get("IKGR_SPECS", ",".join(all_specs)).split(",")

    split = os.environ.get("IKGR_SPLIT", "RS").upper()
    out_path = "run/slice_eval_result.json" if split == "RS" else f"run/slice_eval_{split}_result.json"
    results = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    for name in spec_names:
        model_arg, extra = all_specs[name]
        results.setdefault(name, {})
        if "seeds" not in results[name]:
            results[name]["seeds"] = {}
        for seed in seeds:
            if str(seed) in results[name]["seeds"]:
                print(f"[skip] {name} seed={seed}", flush=True); continue
            print(f"\n===== {name} seed={seed} =====", flush=True)
            pu, pop, tsec, ni = _train_and_collect(model_arg, extra, rb, paths, seed)
            rep = slice_report(pu, pop, ni); rep["train_sec"] = tsec
            results[name]["seeds"][str(seed)] = rep
            results[name]["agg"] = _aggregate(results[name]["seeds"])
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[{name} s{seed}] overall_ndcg@10={rep['overall']['ndcg@10']} "
                  f"tail_recall@10={rep['long_tail']['recall@10']} cov@10={rep['coverage@10']} "
                  f"({tsec}s)", flush=True)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
