# 실험 산출물 경로 정리

## 1. 먼저 읽을 문서

| 상대경로 | 내용 |
|---|---|
| `explain.md` | 현재까지의 전체 진행 상황과 Goodreads 실험 종료 판정 요약 |
| `AGENTS.md` | 가장 자세한 작업 로그, 실행 순서, 실험 결과, 재개 방법 |
| `IKGR_REPORT.md` | IKGR 단독 실험 정리, 버그 진단, KG/meta-KG ablation |
| `DYNLLM_REPORT.md` | DynLLM 단계 정리, 시간순 split, recency 동적 프로필, attention negative result |
| `CORONA_REPORT.md` | CORONA 단계 정리, late-fusion, 후보생성, diversity-first 결과 |
| `DYNLLM_INTEGRATION.md` | DynLLM을 이 레포 구조에 맞게 이식한 설계 문서 |
| `CORONA_INTEGRATION.md` | CORONA를 이 레포 구조에 맞게 이식한 설계 문서 |

## 2. 최종/핵심 결과 JSON

| 상대경로 | 실험 |
|---|---|
| `run_k30/slice_eval_TO_result.json` | k=30 Goodreads Children, 시간순 per-user split(TO), 최종 long-tail 실험 |
| `run_k30/slice_eval_TO_GLOBAL_result.json` | k=30 Goodreads Children, global temporal split(TO_GLOBAL), 최종 cold-start 실험 + soft-rerank low-lambda 결과 |
| `run_k50/slice_eval_TO_result.json` | k=50 시간순 split 결과 |
| `run_k50/slice_eval_TO_GLOBAL_result.json` | k=50 global temporal split 결과 |
| `run/slice_eval_TO_result.json` | k=100 시간순 split에서 IKGR, DynLLM, CORONA ablation 결과 |
| `run/slice_eval_result.json` | k=100 random/per-user split 기반 IKGR multi-seed slice 결과 |
| `run/slice_eval_singleseed_result.json` | k=100 단일 seed slice 평가 결과 |

현재 보고서/발표에서 가장 중요하게 볼 파일은 `run_k30/slice_eval_TO_GLOBAL_result.json`와 `run_k30/slice_eval_TO_result.json`입니다. k=30이 가장 sparse하고 cold-start/long-tail 주장이 가장 잘 드러납니다.

## 3. IKGR 관련 산출물

| 상대경로 | 내용 |
|---|---|
| `run/ikgr_baseline_result.json` | 초기 IKGR baseline 결과, 구버전 성능 진단용 |
| `run/ikgr_kgon_result.json` | k=100 IKGR KG-on step3 결과 |
| `run/ikgr_kgoff_result.json` | k=100 IKGR KG-off, 사실상 MF sanity check 결과 |
| `run/baselines_result.json` | Pop/BPR/LightGCN baseline 결과 |
| `run/slice_eval_result.json` | k=100 IKGR meta-KG/full hetero multi-seed 결과 |
| `IKGR_REPORT.md` | 위 결과의 해석과 표 정리 |

관련 코드:

| 상대경로 | 내용 |
|---|---|
| `step1.py` | LLM intent 추출 |
| `step2.py` | RAG 기반 related intent 확장 |
| `build_kg.py` | intent KG pack 생성 |
| `build_meta_kg.py` | author/publisher/shelf 기반 meta-KG 생성 |
| `build_intent_banks.py` | user/item intent embedding bank 생성 |
| `convert_to_recbole_atomic.py` | RecBole atomic format 변환 |
| `step3.py` | IKGR 학습/평가 |
| `eval_slices.py` | slice 평가, TO/TO_GLOBAL split, ablation 실행 |
| `ikgr_core/model_ikgr.py` | IKGR/DynLLM/CORONA 변형이 들어간 핵심 모델 |

## 4. DynLLM 관련 산출물

| 상대경로 | 내용 |
|---|---|
| `DYNLLM_REPORT.md` | DynLLM 실험 요약 |
| `DYNLLM_INTEGRATION.md` | DynLLM 통합 설계 |
| `run/slice_eval_TO_result.json` | k=100 TO split에서 `IKGR_dyn`, attention ablation 등 포함 |
| `run/slice_dyn.log` | DynLLM recency 실험 로그 |
| `run/slice_attn.log` | multi-facet attention 실험 로그 |
| `dynmLLM/` | 참고한 DynLLM 원본/참조 코드 |

주의: 현재 DynLLM 구현은 LLM-heavy dynamic profile 완성형이 아니라, recency 기반 동적 유저 표현을 중심으로 한 검증 버전입니다.

## 5. CORONA 관련 산출물

| 상대경로 | 내용 |
|---|---|
| `CORONA_REPORT.md` | CORONA 실험 요약 |
| `CORONA_INTEGRATION.md` | CORONA 통합 설계 |
| `run/slice_eval_TO_result.json` | k=100 TO split에서 `IKGR_full`, `IKGR_cand`, `IKGR_cand_db` 결과 포함 |
| `run_k30/slice_eval_TO_GLOBAL_result.json` | k=30 TO_GLOBAL에서 `IKGR_cand_db`, `IKGR_rerank_db_rel_l0p0/l0p01/l0p03/l0p05` 결과 포함 |
| `run_k30/determinism_check.json` | 로컬 산출물. `IKGR_dyn` 반복 재현성과 `rerank lambda=0` 대조군 검증 결과 |
| `run_k30/rerank_low_lambda_TO_GLOBAL.stdout.log` | soft-rerank low-lambda 3seed 실행 로그 |
| `run/corona_full.log` | CORONA late-fusion 실험 로그 |
| `ikgr_core/corona_retriever.py` | CORONA 그래프 후보생성 구현 |
| `CORONA-main/` | 참고한 CORONA 원본/참조 코드 |

주의: 현재 CORONA는 전체 정확도 향상보다는 diversity/long-tail 후보생성에서 효과가 나타났고, overall NDCG는 손해가 있었습니다. soft-rerank는 재현성 고정 후 low-lambda까지 확인했지만 성능 개선으로 채택하기 어렵습니다.

## 6. k별 데이터/중간 산출물

아래 파일들은 대용량이라 대부분 git에는 올라가지 않고 로컬 디스크에만 있습니다.

| 상대경로 | 내용 |
|---|---|
| `data/k_core/interactions_k100.csv` | k=100 상호작용 데이터 |
| `data/k_core/interactions_k100_ts.csv` | k=100 timestamp 복원 데이터 |
| `data/k_core/profiles_k100.csv` | k=100 profile cover 파일 |
| `data/k_core/ikgr-custom/` | k=100 RecBole atomic 데이터 |
| `run/step1_intents.csv` | k=100 step1 intent 결과 |
| `run/step2_related_intents.csv` | k=100 step2 related intent 결과 |
| `run/user_bank.pt`, `run/item_bank.pt` | k=100 user/item intent embedding bank |
| `run/kg_pack.pt`, `run/meta_kg_pack.pt` | k=100 intent/meta KG pack |
| `data/kc_k50/` | k=50 데이터, timestamp, RecBole 변환 산출물 |
| `run_k50/` | k=50 intent, bank, KG, 평가 결과 |
| `data/kc_k30/` | k=30 데이터, timestamp, RecBole 변환 산출물 |
| `run_k30/` | k=30 intent, bank, KG, 평가 결과 |

## 7. 재현/실행 관련 파일

| 상대경로 | 내용 |
|---|---|
| `run_pipeline.py` | k별 전체 파이프라인 실행 오케스트레이터 |
| `run_pipeline_auto.py` | 자동 실행 보조 스크립트 |
| `apply_k_core.py` | k-core 데이터 생성 |
| `goodreads_preprocess.py` | Goodreads 원본 전처리 |
| `add_timestamps.py` | 기존 k-core 데이터에 timestamp backfill |
| `make_temporal_inter.py` | timestamp 포함 RecBole inter 생성 |
| `run_baselines.py` | Pop/BPR/LightGCN baseline 실행 |
| `run_rerank_orchestrator.py` | soft-rerank validation/grid 자동 실행 보조 스크립트 |
| `verify_rerank_baseline.py` | 기존 canonical 결과와 현재 `IKGR_dyn`/rerank 대조군 차이를 점검하는 보조 스크립트 |
| `verify_deterministic_rerank.py` | canonical 결과 JSON을 건드리지 않고 deterministic 학습 및 `lambda=0` 대조군을 검증하는 스크립트 |
| `config.yaml` | 기본 k=100 설정 |
| `run_k50/config.k50.yaml` | k=50 실행 설정 |
| `run_k30/config.k30.yaml` | k=30 실행 설정 |

## 8. 동료에게 추천하는 확인 순서

1. `explain.md`로 전체 결론을 먼저 확인한다.
2. 세부 모델별 해석은 `IKGR_REPORT.md`, `DYNLLM_REPORT.md`, `CORONA_REPORT.md`를 본다.
3. 최종 수치는 `run_k30/slice_eval_TO_result.json`와 `run_k30/slice_eval_TO_GLOBAL_result.json`를 본다.
4. k=100에서 어떤 ablation을 거쳐 지금 결론이 나왔는지는 `run/slice_eval_TO_result.json`와 `run/slice_eval_result.json`를 본다.
5. 재현이 필요하면 `AGENTS.md`의 최신 k=30 섹션과 `run_pipeline.py` 사용법을 따른다.
