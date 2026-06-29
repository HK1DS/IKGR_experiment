#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end IKGR -> DynLLM -> CORONA pipeline orchestrator, parameterized by the
k-core value k. Lets us "build once, then sweep k".

Given --k K, it generates an ISOLATED per-k config + artifact layout and runs the
full pipeline so different k values never clobber each other:

  data/kc_k{K}/                        (data + RecBole atomic files)
    interactions_k{K}.csv, profiles_k{K}.csv, interactions_k{K}_ts.csv
    ikgr-custom/ikgr-custom.inter, .kg
  run_k{K}/                            (run config + all artifacts/results)
    config.k{K}.yaml
    step1_intents.csv, step2_related_intents.csv
    intents_emb.npy, intent_vocab.json, intents.ann
    user_bank.pt, item_bank.pt, kg_pack.pt, meta_kg_pack.pt
    step1_cache.json, step2_cache.json
    slice_eval_TO_result.json

Stages (select a subset with --steps):
  A  apply_k_core            free      build k-core data
  B  step1  (LLM)            $$$       exact intent extraction
  C  step2  (LLM + RAG)      $$$       related intent expansion
  D  build banks/KG/meta + convert + timestamps + temporal inter   free
  E  eval_slices (ablation)  free      train + sliced evaluation

The config-reading scripts (step1/2/3, eval_slices, run_baselines) honor the
IKGR_CONFIG env var, which this orchestrator points at the per-k config.

Cost saver: by default the existing base caches (run/step1_cache.json,
run/step2_cache.json) are copied into run_k{K}/ before the LLM steps, so any
profile already extracted at another k is reused for free (cache is keyed by
profile text). Disable with --no-seed-cache.

Examples:
  python run_pipeline.py --k 30                 # full pipeline for k=30
  python run_pipeline.py --k 30 --steps A       # just build the k-core data
  python run_pipeline.py --k 30 --steps DE      # rebuild packs + evaluate (free)
  python run_pipeline.py --k 30 --steps E --specs IKGR_kgoff,IKGR_full_hetero
"""
import argparse, os, sys, subprocess, shutil, time
import yaml

PY = sys.executable


def sh(cmd, env=None, cwd=None):
    """Run a subprocess, streaming output; raise on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=cwd)
    dt = round(time.time() - t0, 1)
    if r.returncode != 0:
        raise SystemExit(f"[FAIL] ({dt}s) exit={r.returncode}: {' '.join(cmd)}")
    print(f"[ok] ({dt}s) {cmd[1] if len(cmd) > 1 else cmd[0]}", flush=True)


def make_config(k, base="config.yaml"):
    """Derive a per-k config from the base config, isolating all paths."""
    cfg = yaml.safe_load(open(base, encoding="utf-8"))
    data_dir = f"data/kc_k{k}"
    run_dir = f"run_k{k}/"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    p = cfg["paths"]
    p["input_csv"] = f"{data_dir}/profiles_k{k}.csv"
    p["inter_file"] = f"{data_dir}/interactions_k{k}.csv"
    p["workdir"] = run_dir
    p["step1_output"] = f"{run_dir}step1_intents.csv"
    p["step2_output"] = f"{run_dir}step2_related_intents.csv"
    p["kg_triples"] = f"{run_dir}kg_triples.txt"
    p["recbole_dump"] = f"{run_dir}recbole/"
    p["user_bank_pt"] = f"{run_dir}user_bank.pt"
    p["item_bank_pt"] = f"{run_dir}item_bank.pt"
    p["kg_pack"] = f"{run_dir}kg_pack.pt"
    p["meta_kg_pack"] = f"{run_dir}meta_kg_pack.pt"

    r = cfg["rag"]
    r["encoding_npy"] = f"{run_dir}intents_emb.npy"
    r["annoy_index"] = f"{run_dir}intents.ann"
    r["vocab_json"] = f"{run_dir}intent_vocab.json"

    cfg_path = f"{run_dir}config.k{k}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[config] wrote {cfg_path}")
    return cfg_path, cfg


def seed_caches(run_dir, base_run="run/"):
    """Copy existing base caches into run_dir to reuse already-extracted intents."""
    for name in ("step1_cache.json", "step2_cache.json"):
        src = os.path.join(base_run, name)
        dst = os.path.join(run_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"[cache] seeded {dst} from {src}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, required=True, help="k-core value")
    ap.add_argument("--steps", default="ABCDE",
                    help="subset of stages to run, e.g. ABCDE / DE / E (default all)")
    ap.add_argument("--profiles_in", default="data/profiles.csv")
    ap.add_argument("--interactions_in", default="data/interactions.csv")
    ap.add_argument("--books_gz", default="goodreads_books_children.json.gz")
    ap.add_argument("--raw_inter_gz", default="goodreads_interactions_children.json.gz")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--dataset", default="ikgr-custom")
    ap.add_argument("--no-seed-cache", action="store_true",
                    help="do not reuse base run/ LLM caches for this k")
    ap.add_argument("--seed-cache-from", default="run/",
                    help="dir to seed LLM caches from (use run_k50/ for k=30 so "
                         "the ~40k already-extracted profiles are reused, not re-paid)")
    # eval (stage E) knobs
    ap.add_argument("--split", default="TO")
    ap.add_argument("--epochs", default="12")
    ap.add_argument("--seeds", default="2020,2021,2022")
    ap.add_argument("--specs", default="IKGR_kgoff,IKGR_full_hetero,IKGR_dyn,IKGR_cand_db,BPR,LightGCN")
    args = ap.parse_args()

    k = args.k
    steps = set(args.steps.upper())
    cfg_path, cfg = make_config(k)
    data_dir = f"data/kc_k{k}"
    run_dir = f"run_k{k}/"
    ds = args.dataset

    # env for config-reading scripts
    env = dict(os.environ)
    env["IKGR_CONFIG"] = cfg_path

    inter_csv = f"{data_dir}/interactions_k{k}.csv"
    inter_ts = f"{data_dir}/interactions_k{k}_ts.csv"
    step2_csv = f"{run_dir}step2_related_intents.csv"
    kg_pack = f"{run_dir}kg_pack.pt"
    ds_dir = f"{data_dir}/{ds}"

    print(f"\n===== run_pipeline k={k} | steps={''.join(sorted(steps))} =====")

    # A) k-core data ------------------------------------------------------------
    if "A" in steps:
        sh([PY, "apply_k_core.py", "--profiles_in", args.profiles_in,
            "--interactions_in", args.interactions_in, "--k", str(k),
            "--out_dir", data_dir], env=env)

    # seed LLM caches before B/C to save cost
    if ("B" in steps or "C" in steps) and not args.no_seed_cache:
        seed_caches(run_dir, args.seed_cache_from)

    # B) step1 exact intents (LLM) ---------------------------------------------
    if "B" in steps:
        print("[cost] stage B (step1) calls the LLM per unique profile. "
              "Lower k => more entities => higher cost.")
        sh([PY, "-u", "step1.py"], env=env)

    # C) step2 related intents (LLM + RAG) -------------------------------------
    if "C" in steps:
        print("[cost] stage C (step2) calls the LLM per row needing expansion.")
        sh([PY, "-u", "step2.py"], env=env)

    # D) banks + KG + meta-KG + RecBole atomic + timestamps + temporal inter ----
    if "D" in steps:
        sh([PY, "build_intent_banks.py", "--step2_csv", step2_csv,
            "--encoder", args.encoder,
            "--user_out", f"{run_dir}user_bank.pt",
            "--item_out", f"{run_dir}item_bank.pt"], env=env)
        sh([PY, "build_kg.py", "--step2_csv", step2_csv,
            "--vocab", f"{run_dir}intent_vocab.json",
            "--emb", f"{run_dir}intents_emb.npy", "--out", kg_pack], env=env)
        sh([PY, "build_meta_kg.py", "--books_gz", args.books_gz,
            "--kg_pack", kg_pack, "--out", f"{run_dir}meta_kg_pack.pt"], env=env)
        sh([PY, "convert_to_recbole_atomic.py", "--interactions", inter_csv,
            "--intents", step2_csv, "--out_dir", data_dir, "--dataset", ds], env=env)
        sh([PY, "add_timestamps.py", "--inter", inter_csv,
            "--raw", args.raw_inter_gz, "--out", inter_ts], env=env)
        sh([PY, "make_temporal_inter.py", "--inter_ts", inter_ts,
            "--out", f"{ds_dir}/{ds}.inter"], env=env)

    # E) ablation evaluation ----------------------------------------------------
    if "E" in steps:
        eenv = dict(env)
        eenv["IKGR_SPLIT"] = args.split
        eenv["IKGR_EPOCHS"] = args.epochs
        eenv["IKGR_SEEDS"] = args.seeds
        eenv["IKGR_SPECS"] = args.specs
        sh([PY, "-u", "eval_slices.py"], env=eenv)
        out = os.path.join(run_dir, "slice_eval_result.json" if args.split == "RS"
                           else f"slice_eval_{args.split}_result.json")
        print(f"\n[done] k={k} results -> {out}")

    print(f"\n===== run_pipeline k={k} complete =====")


if __name__ == "__main__":
    main()
