#!/usr/bin/env python3
"""Run the guarded CORONA reranker experiment without manual supervision.

The runner never continues beyond a failed lambda=0 reproducibility check. It
then runs a primary grid and, only if no accuracy/long-tail balance is found,
tries one conservative lower-lambda grid.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "run_k30"
RESULT_PATH = RUN_DIR / "slice_eval_TO_GLOBAL_result.json"
STATUS_PATH = RUN_DIR / "rerank_orchestrator_status.json"
LOG_PATH = RUN_DIR / "rerank_orchestrator.log"
SUMMARY_PATH = RUN_DIR / "rerank_orchestrator_summary.md"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SPEC = "IKGR_rerank_db_rel"
BASELINE = "IKGR_dyn"
PRIMARY_GRID = [0.0, 0.1, 0.25, 0.5]
FALLBACK_GRID = [0.0, 0.01, 0.03, 0.05]
TOLERANCE = 0.0001
NDCG_TOLERANCE = 0.0005
TAIL_GAIN_MIN = 0.0001


def _stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _status(stage, **extra):
    payload = {"updated_at": _stamp(), "stage": stage, **extra}
    _write_json(STATUS_PATH, payload)
    return payload


def _load_results():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _key(lam):
    return f"{SPEC}_l{str(float(lam)).replace('.', 'p')}"


def _nested(value, *keys):
    for key in keys:
        value = value[key]
    return float(value)


def _run_eval(seeds, grid, stage):
    env = os.environ.copy()
    env.update({
        "IKGR_CONFIG": "run_k30/config.k30.yaml",
        "IKGR_SPLIT": "TO_GLOBAL",
        "IKGR_EPOCHS": "12",
        "IKGR_SEEDS": seeds,
        "IKGR_SPECS": SPEC,
        "IKGR_CORONA_RERANK_GRID": ",".join(str(value) for value in grid),
    })
    _status(stage, seeds=seeds, grid=grid, log=str(LOG_PATH.relative_to(ROOT)))
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_stamp()}] stage={stage} seeds={seeds} grid={grid}\n")
        log.flush()
        completed = subprocess.run([str(PYTHON), "-u", "eval_slices.py"], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[{_stamp()}] stage={stage} exit={completed.returncode}\n")
    return completed.returncode


def _baseline_match(results):
    base = results[BASELINE]["seeds"]["2020"]
    zero = results[_key(0.0)]["seeds"]["2020"]
    fields = {
        "overall.ndcg@10": ("overall", "ndcg@10"),
        "overall.recall@10": ("overall", "recall@10"),
        "tail.recall@10": ("long_tail", "recall@10"),
        "cold0.recall@10": ("cold_abs_buckets", "cold0_train", "recall@10"),
    }
    deltas = {name: round(_nested(zero, *path) - _nested(base, *path), 8)
              for name, path in fields.items()}
    return all(abs(value) <= TOLERANCE for value in deltas.values()), deltas


def _balanced_candidates(results, grid):
    base = results[BASELINE]["agg"]
    base_ndcg = float(base["overall.ndcg@10"]["mean"])
    base_tail = float(base["long_tail.recall@10"]["mean"])
    candidates = []
    for lam in grid:
        if lam == 0.0:
            continue
        agg = results[_key(lam)]["agg"]
        ndcg = float(agg["overall.ndcg@10"]["mean"])
        tail = float(agg["long_tail.recall@10"]["mean"])
        if ndcg >= base_ndcg - NDCG_TOLERANCE and tail >= base_tail + TAIL_GAIN_MIN:
            candidates.append({"lambda": lam, "ndcg@10": ndcg, "tail_recall@10": tail})
    return candidates


def _write_summary(outcome, details):
    lines = ["# CORONA Reranker Automated Run", "", f"- Finished: {_stamp()}", f"- Outcome: `{outcome}`"]
    for key, value in details.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", f"- Status: `{STATUS_PATH.relative_to(ROOT)}`", f"- Log: `{LOG_PATH.relative_to(ROOT)}`"])
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RUN_DIR.mkdir(exist_ok=True)
    if not PYTHON.exists():
        raise RuntimeError(f"Project Python not found: {PYTHON}")
    _status("started", primary_grid=PRIMARY_GRID, fallback_grid=FALLBACK_GRID)

    if _run_eval("2020", PRIMARY_GRID, "validate_seed_2020") != 0:
        _status("failed_validation_command")
        _write_summary("failed_validation_command", {})
        return 1
    results = _load_results()
    matched, deltas = _baseline_match(results)
    if not matched:
        _status("blocked_baseline_mismatch", deltas=deltas)
        _write_summary("blocked_baseline_mismatch", {"deltas": deltas})
        return 2

    if _run_eval("2020,2021,2022", PRIMARY_GRID, "primary_grid") != 0:
        _status("failed_primary_command")
        _write_summary("failed_primary_command", {})
        return 1
    results = _load_results()
    primary = _balanced_candidates(results, PRIMARY_GRID)
    if primary:
        _status("complete_primary", candidates=primary)
        _write_summary("complete_primary", {"candidates": primary})
        return 0

    if _run_eval("2020,2021,2022", FALLBACK_GRID, "fallback_lower_lambda_grid") != 0:
        _status("failed_fallback_command")
        _write_summary("failed_fallback_command", {})
        return 1
    results = _load_results()
    fallback = _balanced_candidates(results, FALLBACK_GRID)
    outcome = "complete_fallback" if fallback else "complete_no_balanced_lambda"
    _status(outcome, candidates=fallback)
    _write_summary(outcome, {"candidates": fallback})
    return 0


if __name__ == "__main__":
    sys.exit(main())
