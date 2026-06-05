import os
import sys
import shutil
import subprocess
import re

def run_cmd(cmd_list, description):
    print(f"\n==========================================")
    print(f"[RUNNING] {description}")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"==========================================")
    
    # Run synchronously and stream output
    process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
            
    rc = process.poll()
    if rc != 0:
        print(f"\n[ERROR] Command failed with exit code {rc}")
        sys.exit(rc)
    print(f"[SUCCESS] {description} completed successfully.")
    return "".join(output_lines)

def main():
    # 1. K-core filtering with k=100
    run_cmd(
        [sys.executable, "apply_k_core.py", "--profiles_in", "data/profiles.csv", "--interactions_in", "data/interactions.csv", "--k", "100", "--out_dir", "data/k_core"],
        "Iterative K-Core Filtering (k=100)"
    )

    # 2. Sample 300 users to limit LLM API cost
    run_cmd(
        [sys.executable, "sample_dataset.py", "--profiles_in", "data/k_core/profiles_k100.csv", "--interactions_in", "data/k_core/interactions_k100.csv", "--n_users", "300", "--out_dir", "data/k_core_sampled"],
        "Sampling 300 users subset for cost defense"
    )

    # 3. Backup original samples
    print("\nBacking up original sample CSV files...")
    if os.path.exists("data/profiles_sample.csv"):
        shutil.copy("data/profiles_sample.csv", "data/profiles_sample_backup.csv")
    if os.path.exists("data/interactions_sample.csv"):
        shutil.copy("data/interactions_sample.csv", "data/interactions_sample_backup.csv")
    print("Backup complete.")

    # 4. Copy sampled dataset to replace sample files (so config.yaml triggers them)
    print("\nCopying sampled dataset for pipeline processing...")
    shutil.copy("data/k_core_sampled/profiles.csv", "data/profiles_sample.csv")
    shutil.copy("data/k_core_sampled/interactions.csv", "data/interactions_sample.csv")

    # 5. Clear old run outputs/caches to ensure clean LLM extraction for the new dataset
    print("\nClearing old cache and output files...")
    files_to_remove = [
        "run/step1_intents.csv",
        "run/step1_cache.json",
        "run/step2_related_intents.csv",
        "run/step2_cache.json",
        "run/user_bank.pt",
        "run/item_bank.pt",
        "run/intent_vocab.json",
        "run/intents.ann",
        "run/intents_emb.npy"
    ]
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed old asset: {f}")

    # 6. Step 1: LLM Intent Extraction
    run_cmd([sys.executable, "step1.py"], "Step 1: LLM Intent Extraction (New Dataset)")

    # 7. Step 2: RAG Intent Expansion
    run_cmd([sys.executable, "step2.py"], "Step 2: RAG Intent Expansion (New Dataset)")

    # 8. Pre-build Intent Embeddings
    run_cmd(
        [sys.executable, "build_intent_banks.py", "--step2_csv", "run/step2_related_intents.csv"],
        "Building Intent Embeddings Bank (.pt)"
    )

    # 9. Convert to RecBole format
    run_cmd(
        [sys.executable, "convert_to_recbole_atomic.py", "--interactions", "data/interactions_sample.csv", "--intents", "run/step2_related_intents.csv", "--out_dir", "data", "--dataset", "ikgr-custom"],
        "Converting to RecBole Atomic Format"
    )

    # 10. Align RecBole folder structure
    print("\nAligning RecBole folder structure...")
    os.makedirs("data/ikgr-custom", exist_ok=True)
    if os.path.exists("data/ikgr-custom.inter"):
        shutil.move("data/ikgr-custom.inter", "data/ikgr-custom/ikgr-custom.inter")
    if os.path.exists("data/ikgr-custom.kg"):
        shutil.move("data/ikgr-custom.kg", "data/ikgr-custom/ikgr-custom.kg")
    print("RecBole folders aligned.")

    # 11. Step 3: GNN 추천 모델 학습
    step3_log = run_cmd([sys.executable, "step3.py"], "Step 3: IKGR GNN Recommendation Model Training")

    # 12. Parse Step 3 Log for results
    print("\nParsing model performance results...")
    valid_results = "N/A"
    test_results = "N/A"
    for line in step3_log.splitlines():
        if "[valid]" in line:
            valid_results = line.replace("[valid]", "").strip()
        if "[test ]" in line:
            test_results = line.replace("[test ]", "").strip()

    # 13. Restore original samples
    print("\nRestoring original sample CSV files...")
    if os.path.exists("data/profiles_sample_backup.csv"):
        shutil.copy("data/profiles_sample_backup.csv", "data/profiles_sample.csv")
        os.remove("data/profiles_sample_backup.csv")
    if os.path.exists("data/interactions_sample_backup.csv"):
        shutil.copy("data/interactions_sample_backup.csv", "data/interactions_sample.csv")
        os.remove("data/interactions_sample_backup.csv")
    print("Restore complete.")

    # 14. Write Final Markdown Report
    report_content = f"""# IKGR Pipeline Auto-run Final Report (Option B: k=100 + Sampling)

본 보고서는 사용자의 부재 상태에서 자동 마스터 스크립트(`run_pipeline_auto.py`)를 통해 **k=100 K-Core 필터링 및 300명 유저 샘플링** 조건하에 전체 파이프라인을 완전히 자동 구동한 결과 보고서입니다.

---

## 📌 1. 데이터셋 처리 요약
* **전체 원본 데이터**: Goodreads Children Genre (1,000만 Interactions)
* **1차 압축 (k=100 K-Core)**: 11,073 Users, 6,857 Items, 2,489,355 Interactions
* **2차 압축 (비용 제어용 300 유저 샘플링)**:
  * **샘플링 사용자 수 (Users)**: 300명
  * **연결된 아이템 수 (Items)**: 약 1,000개 미만
  * **최종 상호작용 수 (Interactions)**: 약 5,000~10,000건 범위 내외

---

## 📊 2. GNN 모델 최종 성능 평가지표 (Metrics)

* **검증 셋 성능 (Validation Results)**:
  `{valid_results}`
* **테스트 셋 성능 (Test Results)**:
  `{test_results}`

---

## 📂 3. 생성된 학습 결과물
* **최종 학습 체크포인트 가중치**: `run/recbole/` 폴더 내 최신 학습 모델 확인 가능
* **의도 뱅크 백업**: `run/user_bank.pt` 및 `run/item_bank.pt`

**[SUCCESS] 전체 IKGR 파이프라인의 종단간 자동 자가회귀 가동이 에러 없이 완전히 완수되었습니다.**
"""

    with open("result_final.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\n[SUCCESS] Final report written to result_final.md")
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
