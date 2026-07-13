#!/usr/bin/env python3
"""Verify that the soft-reranker lambda=0 control matches current IKGR_dyn.

This runner intentionally bypasses eval_slices.main(), so it never invalidates
or overwrites the canonical multi-seed result JSON.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

from eval_slices import _train_and_collect, slice_report
from ikgr_core.model_ikgr import IKGR as IKGRModel


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "run_k30"
RESULT_PATH = RUN_DIR / "slice_eval_TO_GLOBAL_result.json"
OUT_PATH = RUN_DIR / "rerank_repro_check.json"
SEED = 2020
TOLERANCE = 0.0001


def _current_dyn_extra(paths):
    return {
        "use_kg": True,
        "kg_pack_path": str((ROOT / paths["kg_pack"]).resolve()),
        "kg_layers": 1,
        "kg_cap": 32,
        "intent_learnable": False,
        "use_meta_kg": True,
        "meta_kg_path": str((ROOT / paths["meta_kg_pack"]).resolve()),
        "use_dynamic": True,
    }


def _metric_view(report):
    return {
        "overall.ndcg@10": float(report["overall"]["ndcg@10"]),
        "overall.recall@10": float(report["overall"]["recall@10"]),
        "tail.recall@10": float(report["long_tail"]["recall@10"]),
        "cold0.recall@10": float(report["cold_abs_buckets"]["cold0_train"]["recall@10"]),
    }


def _compare(left, right):
    return {key: round(left[key] - right[key], 8) for key in left}


def _run_dyn(paths, rb):
    per_user_by_variant, item_pop, train_sec, n_items, _ = _train_and_collect(
        IKGRModel, _current_dyn_extra(paths), rb, paths, SEED
    )
    per_user = per_user_by_variant[None]
    report = slice_report(per_user, item_pop, n_items)
    report["train_sec"] = train_sec
    return report


def _write(payload):
    temp = OUT_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUT_PATH)


def main():
    os.environ.update({
        "IKGR_CONFIG": "run_k30/config.k30.yaml",
        "IKGR_SPLIT": "TO_GLOBAL",
        "IKGR_EPOCHS": "12",
    })
    cfg = yaml.safe_load((ROOT / os.environ["IKGR_CONFIG"]).read_text(encoding="utf-8"))
    paths, rb = cfg["paths"], cfg["recbole"]
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    zero = results["IKGR_rerank_db_rel_l0p0"]["seeds"][str(SEED)]
    zero_metrics = _metric_view(zero)

    first = _run_dyn(paths, rb)
    first_metrics = _metric_view(first)
    zero_delta = _compare(first_metrics, zero_metrics)
    payload = {
        "seed": SEED,
        "tolerance": TOLERANCE,
        "rerank_zero": zero_metrics,
        "current_dyn_run_1": first_metrics,
        "rerank_zero_delta": zero_delta,
        "first_train_sec": first["train_sec"],
    }
    if all(abs(value) <= TOLERANCE for value in zero_delta.values()):
        payload["outcome"] = "pass_current_dyn_matches_rerank_zero"
        _write(payload)
        return 0

    second = _run_dyn(paths, rb)
    second_metrics = _metric_view(second)
    repeat_delta = _compare(second_metrics, first_metrics)
    payload.update({
        "current_dyn_run_2": second_metrics,
        "repeat_delta": repeat_delta,
        "second_train_sec": second["train_sec"],
    })
    if all(abs(value) <= TOLERANCE for value in repeat_delta.values()):
        payload["outcome"] = "blocked_reranker_or_eval_path_diff"
        _write(payload)
        return 2
    payload["outcome"] = "blocked_training_nondeterminism"
    _write(payload)
    return 3


if __name__ == "__main__":
    sys.exit(main())
