#!/usr/bin/env python3
"""Check deterministic training and the soft-rerank lambda=0 control.

This script does not touch the canonical slice-eval JSON. It directly trains:
1. IKGR_dyn with the requested seed.
2. IKGR_dyn again with the same seed.
3. IKGR_rerank_db_rel with only lambda=0.

The first comparison detects training nondeterminism. The second verifies that
lambda=0 is a true no-op control for the rerank path.
"""
import json
import os
import time
from pathlib import Path

import yaml

from eval_slices import _train_and_collect, slice_report
from ikgr_core.model_ikgr import IKGR as IKGRModel


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "run_k30"
OUT_PATH = RUN_DIR / "determinism_check.json"
SEED = int(os.environ.get("IKGR_DETERMINISM_SEED", "2020"))
TOLERANCE = float(os.environ.get("IKGR_DETERMINISM_TOL", "0.0001"))


def _extra(paths, rerank=False):
    extra = {
        "use_kg": True,
        "kg_pack_path": str((ROOT / paths["kg_pack"]).resolve()),
        "kg_layers": 1,
        "kg_cap": 32,
        "intent_learnable": False,
        "use_meta_kg": True,
        "meta_kg_path": str((ROOT / paths["meta_kg_pack"]).resolve()),
        "use_dynamic": True,
    }
    if rerank:
        extra.update({
            "corona_rerank_grid": [0.0],
            "corona_cf": False,
            "corona_idf": True,
            "corona_popnorm": 0.5,
        })
    return extra


def _metric_view(report):
    return {
        "overall.ndcg@10": float(report["overall"]["ndcg@10"]),
        "overall.recall@10": float(report["overall"]["recall@10"]),
        "tail.recall@10": float(report["long_tail"]["recall@10"]),
        "cold0.recall@10": float(report["cold_abs_buckets"]["cold0_train"]["recall@10"]),
    }


def _compare(left, right):
    return {key: round(left[key] - right[key], 8) for key in left}


def _run(label, paths, rb, rerank=False):
    started = time.time()
    per_user_by_variant, item_pop, train_sec, n_items, _ = _train_and_collect(
        IKGRModel, _extra(paths, rerank=rerank), rb, paths, SEED
    )
    variant = 0.0 if rerank else None
    report = slice_report(per_user_by_variant[variant], item_pop, n_items)
    metrics = _metric_view(report)
    return {
        "label": label,
        "metrics": metrics,
        "train_sec": train_sec,
        "wall_sec": round(time.time() - started, 1),
    }


def _write(payload):
    RUN_DIR.mkdir(exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main():
    os.environ.setdefault("IKGR_CONFIG", "run_k30/config.k30.yaml")
    os.environ.setdefault("IKGR_SPLIT", "TO_GLOBAL")
    os.environ.setdefault("IKGR_EPOCHS", "12")
    cfg = yaml.safe_load((ROOT / os.environ["IKGR_CONFIG"]).read_text(encoding="utf-8"))
    paths, rb = cfg["paths"], cfg["recbole"]

    payload = {
        "seed": SEED,
        "tolerance": TOLERANCE,
        "config": os.environ["IKGR_CONFIG"],
        "split": os.environ["IKGR_SPLIT"],
        "epochs": int(os.environ["IKGR_EPOCHS"]),
        "runs": [],
    }

    dyn1 = _run("IKGR_dyn_run_1", paths, rb, rerank=False)
    payload["runs"].append(dyn1)
    _write(payload)

    dyn2 = _run("IKGR_dyn_run_2", paths, rb, rerank=False)
    payload["runs"].append(dyn2)
    repeat_delta = _compare(dyn2["metrics"], dyn1["metrics"])
    payload["dyn_repeat_delta"] = repeat_delta
    _write(payload)

    if any(abs(value) > TOLERANCE for value in repeat_delta.values()):
        payload["outcome"] = "blocked_training_nondeterminism"
        _write(payload)
        return 3

    zero = _run("IKGR_rerank_lambda_0", paths, rb, rerank=True)
    payload["runs"].append(zero)
    zero_delta = _compare(zero["metrics"], dyn1["metrics"])
    payload["rerank_zero_delta"] = zero_delta
    payload["outcome"] = (
        "pass_deterministic_rerank_zero"
        if all(abs(value) <= TOLERANCE for value in zero_delta.values())
        else "blocked_rerank_zero_mismatch"
    )
    _write(payload)
    return 0 if payload["outcome"].startswith("pass_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
