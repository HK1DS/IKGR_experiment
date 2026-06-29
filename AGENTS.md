# IKGR Pipeline Progress & Execution Guide

이 문서는 IKGR 파이프라인의 진행 상황 및 새 세션/새 환경에서 이어서 진행할 때 참고하는 실행 가이드입니다.
**(최종 갱신: Step A(k-core 생성) + Step B(step1 의도 추출) 완료, Step C(step2 RAG 확장) 진행 대기 시점)**

### ✅ 현재 진행 현황 한눈에 보기
| 단계 | 내용 | 상태 |
|------|------|------|
| Step A | k=100 K-core 실데이터 생성 | ✅ 완료 (`data/k_core/interactions_k100.csv`, `profiles_k100.csv`) |
| Step B | step1 — LLM 의도 추출 | ✅ 완료 (`run/step1_intents.csv`, **11,073행 전부 채워짐**, exact intent 빈 셀 0) |
| Step C | step2 — RAG 의도 확장 | ✅ **완료** (`run/step2_related_intents.csv`, 11,073행 전부 채워짐 / related 평균 user 14.15·item 14.58개) |
| Step D | 임베딩 뱅크 + RecBole 포맷 변환 | ✅ **완료** (`run/user_bank.pt` 11,073×768, `run/item_bank.pt` 6,857×768, `data/k_core/ikgr-custom/{inter,kg}` 배치 완료) |
| Step E | step3 — IKGR GNN 학습/평가 | ✅ **완료** (baseline 기록: `run/ikgr_baseline_result.json` / test NDCG@10 0.045, Hit@10 0.296, MRR@10 0.125) |
| 참조 baseline | Pop/BPR/LightGCN (동일 split) | ✅ **완료** (`run/baselines_result.json`, `run_baselines.py`) — 아래 ⚠️ 중요 발견 참고 |
| 다음 | **IKGR 스코어/KG 수정 (최우선)** → 슬라이스 평가 → DynLLM | ⬜ 대기 |

### ✅ 해결됨 — IKGR 재설계로 0.045 → 0.29대 (BPR/LightGCN 동급). 단 KG는 overall에선 도움 안 됨
**구버전 IKGR 스코어 버그(확정):** `score = y_ui(intent max-cos, 고정 휴리스틱 0.5~1.0) + 0.1·z_ui(임베딩)`에서 고정 휴리스틱이 랭킹을 지배 + penalty cliff(0.9)로 유저당 수백~1000+ 아이템이 top에 동점 → top-10 사실상 랜덤 → Pop보다 낮음. (진단 수치: y_val 평균0.804, 22.5%>0.9; y_ui 분산0.257 ≫ λ상한0.1)

**재설계(현재 코드):** 고정 휴리스틱 제거. 공유 **학습 가능 intent 노드**(`run/kg_pack.pt`, mpnet 초기화+학습 투영) + 희소 user/item→intent 인접행렬 + 벡터화 1-layer 전파(`u'=e_u+α·Â_u·E_int`) + **내적 BPR** 학습. `use_kg` 토글로 ablation. 부수효과로 평가도 내적이라 수초로 빨라짐(기존 4.6분 병목 해소).

**전체 비교 (동일 split·seed=2020·emb=512, test):**
| 모델 | NDCG@10 | Recall@10 | MRR@10 | Hit@10 |
|---|---|---|---|---|
| IKGR 구버전(KG-off 휴리스틱, 버그) | 0.045 | 0.021 | 0.125 | 0.296 |
| Pop | 0.154 | 0.077 | 0.363 | 0.647 |
| **IKGR 신버전 KG-off (=MF)** | **0.294** | 0.153 | 0.565 | 0.879 |
| BPR | 0.295 | 0.154 | 0.566 | 0.882 |
| LightGCN | 0.296 | 0.148 | 0.563 | 0.867 |
| **IKGR 신버전 KG-on** | **0.281** | 0.148 | 0.547 | 0.872 |

* KG-off(MF) 0.294 ≈ BPR 0.295 → 구현 정상 sanity check 통과.
* **KG-on(0.281) < KG-off(0.294)**: dense k=100에선 intent KG 전파가 overall을 살짝 떨어뜨림(일반적 intent 평균 전파가 강한 CF 신호를 희석). **예상된 결과** — intent/KG 가치는 overall이 아니라 cold-start/long-tail 슬라이스에서 봐야 함.
* 결과 파일: `run/ikgr_kgon_result.json`, `run/ikgr_kgoff_result.json`. 재현: `IKGR_USE_KG=1|0 python step3.py` (KG는 `python build_kg.py`로 `run/kg_pack.pt` 선생성 필요).

### ✅ 슬라이스 평가 완료 (`eval_slices.py` → `run/slice_eval_result.json`)
동일 split·seed·emb=512. long-tail = 인기 하위 80% 아이템(컷 pop 384.8, 6858중 ~5487개). cold = 유저 활동도(마스킹 히스토리 수) 분위수 버킷.

| 모델 | overall NDCG@10 | tail Recall@10 | tail Recall@30 |
|---|---|---|---|
| IKGR KG-on | 0.274 | **0.0734** | **0.1415** |
| IKGR KG-off | 0.293 | 0.0704 | 0.1346 |
| BPR | 0.294 | 0.0581 | 0.1160 |
| LightGCN | 0.295 | 0.0450 | 0.0934 |

**발견 1 (긍정):** long-tail Recall에서 **KG-on > KG-off > BPR > LightGCN** (순서 깨끗, @10·@30 모두). KG-on tail Recall@10은 LightGCN 대비 +63%, BPR +26%, KG-off +4~5%. → "intent KG가 long-tail/niche 추천을 돕는다" 주장 성립. overall 최강(LightGCN)이 tail 최약 = 정확도-꼬리 트레이드오프.
**발견 2 (제약):** cold-start는 **이 데이터로 측정 불가** — 활동도 Q1(가장 cold) 컷이 105 인터랙션 → k=100 코어가 진짜 cold 유저를 전부 제거함. cold 버킷에선 KG-on이 KG-off 못 이김(cold가 아니므로).

### ⚠️ 다중 시드 robustness 결과 (12 epochs, seeds 2020/2021/2022) — 단일시드 long-tail 우위는 robust하지 않았음
`eval_slices.py` (multi-seed, mean±std, coverage/novelty 추가). 모델 강화: intent 노드를 독립 학습(`intent_learnable`) + multi-layer(`kg_layers`) + 미니배치 gather(`kg_cap`) 지원. **주의: 학습 epochs=12 (이전 단일시드 표는 20) → 수치 직접 비교 불가, 이 표 내부만 비교.**

| 모델 | overall NDCG@10 | tail Recall@10 | tail Recall@30 | coverage@10 | novelty |
|---|---|---|---|---|---|
| IKGR KG-off (MF) | 0.2922±.0009 | 0.0597±.0008 | 0.1180±.0010 | 0.640 | 10.68 |
| IKGR KG-on L1(학습노드) | 0.2423±.0030 | 0.0585±.0104 | 0.1140±.0155 | 0.738±.087 | 10.97 |
| BPR | 0.2935±.0021 | 0.0600±.0015 | 0.1186±.0013 | 0.643 | 10.67 |
| LightGCN(12ep, 저평가) | 0.2585±.0017 | 0.0276±.0005 | 0.0566±.0006 | 0.303 | 9.94 |

**핵심 교훈:** 단일시드에서 본 KG-on long-tail 우위(0.0734)는 **재현 안 됨**. 3시드 평균 tail Recall@10은 KG-on 0.0585 ≈ KG-off/BPR 0.0597/0.0600 (동률), 게다가 **분산 ±18%로 매우 불안정**(seed 2021은 0.044로 붕괴). 학습 가능 intent 노드(27M 파라미터)의 instability가 원인 추정.
**유일하게 일관된 KG 효과 = coverage/novelty↑** (cov 0.738 vs 0.640, +15%; LightGCN은 0.303으로 popularity bias 확정). 단 이것도 분산 큼. **즉 현 IKGR의 검증된 기여는 "정확도"가 아니라 "추천 다양성/coverage"** (overall 정확도는 희생).

**다음 후보:**
1. **frozen-proj 변형 multi-seed** (`intent_learnable=False`) — instability가 학습노드 탓인지 검증. 커밋된 단일시드 0.0734는 frozen-proj였음 → 더 안정적일 가능성.
2. α/reg/epochs 안정화, kg_layers=2 (smoke에서 cov 0.39↑ but 느림).
3. 주장 방향을 "diversity-aware" 쪽으로 재정렬하거나, sparse 코어(k=30)로 가서 정확도 우위 재시도.

### ✅✅ IKGR 최종 결론 (frozen-proj multi-seed 완료 → IKGR 단계 종료)
frozen-proj(`intent_learnable=False`) 3시드 결과 (12ep):

| 변형 (12ep, 3seed) | overall NDCG@10 | tail Recall@10 | cov@10 | 안정성 |
|---|---|---|---|---|
| KG-off (MF) | 0.2922±.001 | 0.0597±.001 | 0.640 | 안정 |
| KG-on L1 frozen | 0.2782±.0002 | 0.0580±.0002 | 0.668±.004 | 매우 안정 |
| KG-on L1 learnable | 0.2423±.003 | 0.0585±.0104 | 0.738±.087 | 불안정 |
| BPR | 0.2935±.002 | 0.0600±.001 | 0.643 | 안정 |

**판정 1:** instability 원인 = 학습 가능 intent 노드(27M 파라미터) 확정. frozen은 분산 ~0.
**판정 2 (IKGR 핵심 결론):** **어느 KG 변형도 long-tail "정확도(Recall)"를 plain MF 대비 robust하게 개선하지 못함** (frozen 0.0580 ≤ MF 0.0597). KG의 **유일하게 일관된 기여 = 추천 다양성/coverage 증가(frozen +4%, learnable +15%)이며 overall 정확도는 희생**.
**판정 3 (epoch 민감도 단서):** 커밋된 단일시드 20ep frozen은 tail 0.0734 > kgoff(20ep) 0.0704 (+4%)였으나 12ep에선 사라짐 → KG의 미약한 tail 이득은 더 긴 학습에서만, 그것도 ~4%로 작음.

**→ IKGR 단계 종료(DynLLM 진입 전 정지).** 정직한 IKGR 스토리: "intent-KG는 dense 데이터에서 정확도 SOTA가 아니라 **diversity/coverage 트레이드오프**를 제공; long-tail 정확도 이득은 작고 조건부(긴 학습)이며, cold-start 검증엔 sparse 코어(k=30) 필요." DynLLM은 별도 scope.

### (이전 기록) 참조 baseline 비교 — 현재는 위 표로 대체됨
| 모델 | NDCG@10 | Recall@10 | MRR@10 | Hit@10 | 학습시간 |
|---|---|---|---|---|---|
| **IKGR (KG off)** | **0.045** | 0.021 | 0.125 | 0.296 | ~30분 |
| Pop | 0.154 | 0.077 | 0.363 | 0.647 | 27초 |
| BPR | 0.295 | 0.154 | 0.566 | 0.882 | 140초 |
| LightGCN | 0.296 | 0.148 | 0.563 | 0.867 | 227초 |

**진단:**
1. IKGR(KG off)이 Pop보다도 낮음 = **버그 수준**. 원인 가설: 스코어 `y_ui(intent max-cos, 고정 휴리스틱) + λ·z_ui(임베딩)`에서 **학습되는 건 임베딩뿐인데 λ=0.1이라 고정 휴리스틱 y_ui가 랭킹을 지배** → BPR loss로 배우는 부분이 무력화. KG도 미사용 상태. **즉 아직 IKGR을 제대로 테스트한 적이 없음.**
2. **싸움터 문제**: k=100은 dense(유저당 평균 225 인터랙션)라 CF(BPR/LightGCN)가 압도적. LLM-intent/KG의 강점은 cold-start/long-tail/sparse 구간 → **overall NDCG로 dense에서 붙는 건 CF 홈그라운드**. 프레임워크 강점은 슬라이스 평가에서 봐야 함(졸업작품 주장도 cold-start/long-tail).
3. **다음 작업 우선순위**: (1) IKGR 스코어/KG 수정해 최소 Pop·가능하면 BPR 근처까지 끌어올리기(선행 필수) → (2) cold-start 유저/long-tail 아이템 슬라이스 평가 틀 → (3) 필요시 k=20/30 코어 재고 → 이후 DynLLM/CORONA.

**참조 baseline 재현:** `python run_baselines.py` (Pop/BPR/LightGCN, `run/recbole_baselines/`에 격리 빌드). RecBole 1.2.0+scipy 1.17 호환 위해 `dok_matrix._update = dict.update` monkeypatch 포함. Pop은 best_valid_result None이라 None-안전 처리됨.

---

## 0. 프로젝트 큰 그림 (졸업작품 목표)

학부 졸업작품. 최종 목표는 **3단 통합 프레임워크가 개별 프레임워크보다 낫다는 것을 보이는 것**:

1. **IKGR** — LLM으로 의도(intent) 추출 → intent-aware KG 구축 *(이 레포에 구현되어 있는 부분)*
2. **DynLLM** — multi-faceted 유저 프로필 생성 → KG 엣지 가중치 동적 갱신 *(미구현)*
3. **CORONA** — 동적 KG 탐색 → 후보 생성 → LLM 필터링 → 가중합 랭킹 *(미구현)*

→ ablation: `Full(1+2+3)` vs `각 컴포넌트 단독`. cold-start / long-tail 슬라이스에서 우위를 보이는 것이 핵심.
→ 평가 기준은 SOTA가 아니라 "동작하고, 신뢰 가능하고, arXiv 수준으로 쓸 수 있는" 최소 실행 가능 버전.

---

## 1. 확정된 결정사항

### 데이터셋: Goodreads Children, **k=100 K-core (진짜 데이터)**
* 샘플 csv가 아니라 원본 `data/profiles.csv`(37GB) / `data/interactions.csv`(1,005만 행)에 k-core 필터 적용.
* k=100 결과: **유저 11,073 / 아이템 6,857 / 인터랙션 2,489,355** (밀도 ~363 인터랙션/아이템).
* 이전 에이전트가 만든 임의 샘플(50유저, 1,000유저)은 폐기. 표준 k-core라 재현 가능.

### LLM provider: Luxia Cloud (gpt-4o-mini 브리지)
* `config.yaml`의 `llm` 섹션이 Luxia로 설정됨. `.env`에 `LUXIA_API_KEY` 세팅 완료(확인됨).
* **남은 크레딧: $59.**

### 비용 현실 (중요 — 레포의 기존 추정은 ~70배 부풀려져 있었음)
* gpt-4o-mini 실단가: 입력 $0.15 / 출력 $0.60 per 1M tokens → **호출당 약 $0.0002~0.0003**.
* `run/k_core_analyzer.py` / `result_example.md`의 "$0.015/call"은 **틀린 값**. 참고하지 말 것.
* k=100 비용 추정:
  * IKGR만(step1+step2, ~36K 호출): **~$7~11**
  * 통합 프레임워크 전체(IKGR+DynLLM+CORONA+디버깅 마진, ~87K 호출): **~$19** → $59 안에서 여유 있음.
* 대안: long-tail을 더 보존하려면 k=20/30 코어가 좋지만 비용↑($120~150). 이 경우 **Qwen-turbo(3배 저렴, 무제한 종량제)** 또는 DeepSeek로 전환 권장. `LocalLLM`이 이미 OpenAI 호환 provider를 지원하므로 `config.yaml`+`.env`만 수정하면 됨.

---

## 2. 이번 세션에서 해결한 잔존 문제 (수정 완료, 저장됨)

1. **`step2.py` `p_rel` 미정의 버그** → `load_prompt()` 추가 + `p_rel = load_prompt("prompts/step2_related.txt")` 로딩 추가. (이전엔 step2 실행 시 즉시 `NameError`)
2. **`apply_k_core.py`가 37GB profiles.csv를 통째로 `pd.read_csv`** → OOM. **메모리 안전 버전으로 재작성**:
   * interactions만 메모리 로드 후 k-core 필터.
   * profiles.csv는 **청크 스트리밍**으로 k-core 엔티티의 프로필만 수집.
   * 모든 유저·아이템이 1회 이상 등장하는 **작은 "커버" 프로필 파일**을 출력 → 다운스트림(step1/step2/banks/convert) 메모리 문제 제거. (LLM은 어차피 고유 프로필당 1회만 호출하므로 손실 없음)
3. **stale 산출물 정리** → `run/_backup_old_sample/`로 이동(삭제 아님):
   * `run/recbole/ikgr-custom-Dataset.pth` (RecBole가 캐시된 옛 2,138행 데이터셋을 재로딩하는 문제), 옛 체크포인트, 옛 `data/ikgr-custom/*.inter/.kg`, 옛 `run/step1_cache.json`(53유저).
4. **`config.yaml` 경로** → `input_csv: data/k_core/profiles_k100.csv`, `inter_file: data/k_core/interactions_k100.csv`로 교체.
5. **(step2 실행 중 발견·수정) Windows Annoy 네이티브 크래시** → `ikgr_core/rag.py` 수정.
   * 증상: step2가 vocab 임베딩(`run/intents_emb.npy`, 53,190×768) 저장 직후 **조용히 종료**(파이썬 트레이스백 없음, 로그 빈 상태). 원인은 `AnnoyIndex.build(50)`에서 네이티브 세그폴트.
   * 격리 재현: npy 로드 후 `ann.build(50)` 호출 시 "building 50 trees..." 출력 후 Exit 1로 사망(예외 없음) → 네이티브 크래시 확정.
   * 수정: `rag.py`가 **win32에서는 자동으로 sklearn `NearestNeighbors`(brute, cosine) 백엔드를 사용**하도록 변경. `IKGR_FORCE_ANNOY=1` 설정 시에만 Annoy 강제. (sklearn 경로는 기존에 폴백 코드로 존재했으나 Annoy import 성공 시 안 타던 것을 플랫폼 기준으로 강제.)
   * 검증: sklearn 경로로 build_ann 2.8초, 쿼리당 ~135ms. step2 정상 진입 확인(`Pending RAG+LLM calls: 15198`, cache 누적).
   * ⚠️ 디버깅 팁: 백그라운드 프로세스 출력이 PowerShell `*>` 리다이렉트로 버퍼링돼 안 보일 수 있음. `python -u ... 2>&1 | Tee-Object -FilePath run\step2_run.log`가 실시간 캡처에 유리. 또한 process tool의 start가 동일 명령을 **재사용(isReused)** 하면 죽은 터미널에 붙어 새 실행이 안 될 수 있으니, 재실행 시 stop 후 새 명령으로 시작할 것.

---

## 3. 재개 순서 (여기서부터 진행)

### Step A. k-core 실데이터 생성 (무료, API 미사용, ~10~30분) — ✅ **완료**
```bash
python apply_k_core.py --profiles_in data/profiles.csv --interactions_in data/interactions.csv --k 100 --out_dir data/k_core
```
* 산출물: `data/k_core/interactions_k100.csv`, `data/k_core/profiles_k100.csv`(커버 파일). **생성 확인됨.**

### Step B. Step 1 — LLM 의도 추출 (⚠️ 유료) — ✅ **완료**
```bash
python step1.py
```
* **결과: `run/step1_intents.csv` 생성 완료. 총 11,073행, `user_profile`/`item_profile`/`user_intents_exact`/`item_intents_exact` 네 컬럼 모두 빈 셀 0개로 검증됨.** 캐시는 `run/step1_cache.json`에 적재됨.
* 고유 프로필 dedup 후 약 15K 호출(유저 8.3K + 아이템 6.8K). 캐시(`run/step1_cache.json`)로 중단/재개 가능.
* **동시성(8 워커, `IKGR_STEP1_WORKERS`로 조절) + 락 기반 thread-safe 캐시 + 원자적 저장 적용됨.**
* **LLM 출력 파싱 견고화**: gpt-4o-mini가 ```` ```python ```` 코드펜스로 감싸 출력하므로, 단순 `startswith("[")` 대신 `parse_intent_list()`로 펜스 제거 후 리스트 추출. (이 수정 없으면 의도가 대량 누락됨 — 검증 완료)

### Step C. Step 2 — RAG 의도 확장 (⚠️ 유료) — ✅ **완료**
```bash
python step2.py
```
* **결과: `run/step2_related_intents.csv` 생성 완료. 11,073행 전부 `user_intents_related`/`item_intents_related` 채워짐 (평균 user 14.15개, item 14.58개). 총 15,198건 RAG+LLM 호출 약 53분 소요.** 캐시 `run/step2_cache.json`.
* ⚠️ **Windows에서는 Annoy 크래시 때문에 `rag.py`가 sklearn `NearestNeighbors`로 자동 폴백** (아래 Section 2-5 참고). 쿼리당 ~135ms brute-force지만 LLM I/O와 겹쳐 처리돼 문제 없음.
* 입력 `run/step1_intents.csv` → 출력 `run/step2_related_intents.csv`. step1 exact intent로 vocab 새로 구성 후 RAG 후보 → LLM 선택.
* 약 15K 호출. 캐시(`run/step2_cache.json`)로 중단/재개 가능. **이 단계 들어가기 전 사용자 재확인 필요.**
* **step1과 동일하게 동시성(8 워커) + 원자적 캐시 저장 적용됨.** 단, 인코더(SentenceTransformer) thread-safety 문제 때문에 임베딩은 동시 단계 전에 배치로 미리 계산하고, 스레드에서는 ANN 검색(읽기) + LLM 호출만 수행. 워커 수는 `IKGR_STEP2_WORKERS`로 조절.

### Step D. 임베딩 뱅크 + RecBole 포맷 변환 — ✅ **완료**
```bash
python build_intent_banks.py --step2_csv run/step2_related_intents.csv --encoder sentence-transformers/all-mpnet-base-v2 --user_out run/user_bank.pt --item_out run/item_bank.pt
python convert_to_recbole_atomic.py --interactions data/k_core/interactions_k100.csv --intents run/step2_related_intents.csv --out_dir data/k_core --dataset ikgr-custom
```
* **결과: `run/user_bank.pt`(11,073 users×768), `run/item_bank.pt`(6,857 items×768) / `data/k_core/ikgr-custom/ikgr-custom.inter`(112MB), `.kg`(40MB) 생성·이동 완료.**
* RecBole 폴더 정렬 완료: `.inter/.kg`를 `data/k_core/ikgr-custom/`로 이동함 (step3가 `data_path = dirname(inter_file) = data/k_core`에서 `data/k_core/ikgr-custom/ikgr-custom.inter`를 찾음).
* ⚡ **(이번 세션 최적화) `build_intent_banks.py` 재작성**: 기존엔 행마다 `encode()`를 호출(약 22K회, 50만+ 문자열)해 CPU에서 수 시간 소요 예상이었음. → **고유 intent(약 53K개)만 1회 배치 인코딩 후 토큰별로 stack**하도록 변경(출력 포맷·순서 동일). 인코딩 ~14초로 단축. 진행바(`show_progress_bar=True`) 추가.

### Step E. Step 3 — IKGR GNN 학습/평가 — ✅ **완료**
```bash
python step3.py
```
* **결과(IKGR 단독 baseline, seed=2020, GPU RTX 3060 Ti 8GB):** `run/ikgr_baseline_result.json`에 저장.
  * test: Recall@10 0.0214 / @30 0.0478, MRR@10 0.1247, NDCG@10 0.0453 / @30 0.0491, Hit@10 0.2958 / @30 0.5153, Precision@10 0.0406.
  * 체크포인트: `run/recbole/IKGR-Jun-07-2026_00-00-46.pth`. variant 라벨 = "intent-bank re-ranking (KGCN/TransH inactive)".
* ⚠️ **이번 세션에서 step3가 그대로는 안 돌아서 수정한 내용 (model_ikgr.py + step3.py):**
  1. **스택 오버플로(exit 0xC00000FD) 수정** — `forward()`의 `_ = self.kgcn(E, R, self.neighbors)`가 노드 17,930개를 파이썬 for 루프로 돌며 `out[v]=...` 인덱스 대입 → ~18k 깊이 autograd 그래프 → 첫 backward에서 Windows 스택(1MB) 초과로 네이티브 크래시(트레이스백 없음). KGCN 결과는 스코어에 안 쓰이는 죽은 값이라 **해당 호출 제거**. (실제 KG neighbor aggregation을 붙일 땐 *벡터화된* KGCN으로 재도입할 것.)
  2. **8GB VRAM OOM 방지** — intent당 최대 111개라 full-sort 평가에서 `[chunk×k×768]` bmm이 청크 하나만 ~17GB. → 노드당 intent **cap=64→32**(평균 25, p99 50이라 손실 미미), 평가 청크 파라미터화(`intent_score_chunk`).
  3. **평가 속도 최적화** — 병목은 full-sort 평가(전체 ~11K 유저×6.8K 아이템 1회 ≈ 4.6~5분). (a) intent bank를 `load_intent_banks`에서 **1회만 L2 정규화**해 저장→`forward`에서 정규화 생략(forward 84→58ms), (b) `intent_score_chunk=8192`가 최적(16384는 VRAM 압박으로 더 느림), (c) `eval_step=5`로 평가 횟수 20→4회 축소. 총 ~30분에 완료.
  4. **재현성**: `seed=2020`, `reproducibility=True`. 결과를 `run/ikgr_baseline_result.json`로 저장하는 로직 추가.
* 참고: RecBole 콘솔 로그가 PowerShell `Tee-Object`로 잘 안 잡힘 → 진행은 `log_tensorboard/<run>/events...` 파일 크기 증가로 모니터링. 최종 산출물은 결과 json + `run/recbole/*.pth`.
* **남은 한계(미해결, 의도적):** KG(.kg)는 여전히 모델에 로드되지 않음 → intent bank 기반 스코어링만 동작. "intent-aware KG" 주장하려면 KGCN neighbor 주입 필요(다음 작업 후보).
* **데이터 품질 메모:** `.inter`의 rating=0 인터랙션이 34%(854,363행, Goodreads "읽을 예정" 미평가). ranking+neg sampling에서 모두 positive로 취급됨. 이번 baseline은 원본 충실하게 0 포함으로 측정. 품질/슬라이스 분석 시 convert 단계에서 `rating>0` 필터 버전도 비교 권장.

---

## 4. 환경 메모
* Windows + cmd 셸. `python`이 직접 안 잡히면 `.venv\Scripts\python.exe` 사용.
* PowerShell에서 `$_` 등 변수 이스케이프가 깨지는 경우가 있으니, 파일 조작은 짧은 Python 스크립트로 처리하는 게 안전.
* `.gitignore`가 `data/`, `run/` 전체를 제외 → 대용량 데이터/산출물은 깃 추적 대상 아님.
* **인코딩 주의**: Windows 기본 인코딩이 `cp949`라 `run/*.json`(예: step1/step2 캐시)을 점검용으로 직접 열 때 `open(path)`만 쓰면 `UnicodeDecodeError`가 난다. 반드시 `open(path, encoding="utf-8")`로 열 것. (파이프라인 코드 자체는 이미 utf-8로 읽고/쓰므로 정상 동작에는 영향 없음 — 디버깅 스크립트에서만 주의.)

---

### ✅✅✅ IKGR 완성 — 이종 메타데이터 KG로 robust한 long-tail 우위 확보 (반쪽 IKGR 진단 해소)
이전까지 KG는 User/Item/Intent + has_intent만 = 다이어그램의 절반. **Brand(저자/출판사)/Category·Attribute(shelves)** 노드는 `goodreads_books_children.json.gz`에서 **LLM 없이 무료** 추출 가능 → `build_meta_kg.py` 작성, `model_ikgr.py`에 `use_meta_kg` 관계별 전파 추가.

이종 KG 규모(6,857 아이템): author 3,901(edge 10,714) / publisher 744(6,102) / shelf 772(98,779).

**결과 (12ep, 3seed, mean±std) — `run/slice_eval_result.json`:**
| 모델 | overall NDCG@10 | tail Recall@10 | tail Recall@30 | cov@10 |
|---|---|---|---|---|
| KG-off (MF) | 0.2922 | 0.0597 | 0.1180 | 0.640 |
| intent-KG frozen | 0.2782 | 0.0580 | 0.1151 | 0.668 |
| **meta-KG** | 0.2677 | **0.0658±.0015** | **0.1306** | 0.657 |
| full hetero | 0.2683 | 0.0653 | 0.1292 | 0.650 |
| BPR | 0.2935 | 0.0600 | 0.1186 | 0.643 |
| LightGCN | 0.2585 | 0.0276 | 0.0566 | 0.303 |

- **meta-KG tail Recall@10 0.0658 = MF +10%, BPR +10%, intent-only +13%** (robust, 분산 작음). full_hetero ≈ meta_only → driver는 메타데이터(shelf). 트레이드오프 overall -8%.
- **재현성:** eval_slices는 매 실행 from-scratch(이전 런 의존 X). meta-KG는 LLM 무관 완전 결정적. intent-KG만 LLM 출력(저장본 기준 결정적)에 의존.
- 재현: `python build_meta_kg.py` → `IKGR_EPOCHS=12 IKGR_SEEDS=2020,2021,2022 IKGR_SPECS=IKGR_meta_only,IKGR_full_hetero python eval_slices.py`
- **IKGR 단계 (이번엔 진짜) 종료.** 다음: cold-start용 sparse 코어(k=30, Qwen 전환) 또는 DynLLM.

---

### ▶ DynLLM 단계 시작 (Option B) — Step 1: 시간순(TO) split + 재baseline 완료
- 설계: `DYNLLM_INTEGRATION.md` (공식 `dynmLLM/` 이식 대신 우리 RecBole/IKGR 위에 메커니즘 재구현).
- timestamp 복원: `goodreads_preprocess.py`(timestamp 인지화) + `add_timestamps.py`(기존 k-core backfill, 2,489,355행 전부 매칭, 99.96% 유효) → `data/k_core/interactions_k100_ts.csv`.
- `make_temporal_inter.py`: garbage 날짜 [2006,2017]로 clip → `data/k_core/ikgr-custom/ikgr-custom.inter`에 `timestamp:float` 추가(1.6% clip). RS 런 하위호환.
- `eval_slices.py`: `IKGR_SPLIT=TO` 지원(`order=TO`, per-user 시간순 split, 결과 `run/slice_eval_TO_result.json`).

**TO baseline (12ep, 3seed, mean±std):**
| 모델 | overall NDCG@10 | tail Recall@10 | tail Recall@30 | cov@10 |
|---|---|---|---|---|
| MF (kgoff) | 0.0777±.0005 | 0.0089±.0003 | 0.0220 | 0.627 |
| **IKGR (full hetero)** | 0.0696±.0017 | **0.0108±.0007** | **0.0269±.0012** | 0.690 |
| BPR | 0.0778±.0004 | 0.0092±.0004 | 0.0227 | 0.633 |
| LightGCN | 0.0836±.0002 | 0.0052±.0002 | 0.0127 | 0.332 |

- 시간순 예측이라 overall NDCG는 RS(0.29대)보다 크게 낮음(정상). **IKGR long-tail 우위는 TO에서 더 큼**: tail@10 +21% vs MF, +17% vs BPR, +108% vs LightGCN. LightGCN은 overall 1위·tail 최악(popularity bias 심화).
- per-user 시간순 split이라 모든 유저가 train에 등장(진짜 cold-start 유저는 아직 없음 — 글로벌 시간 split은 별도 과제).
- **다음 (Step 2): recency 가중 동적 프로필** — 유저 표현을 최근 상호작용 아이템 facet의 시간감쇠 가중 집계로. `use_dynamic` 토글, IKGR vs IKGR+recency 비교.


### ▶ DynLLM Step 2: recency 동적 프로필 — ablation 완료
모델: `_emb_users`에 recency 가중 동적 항 추가 (유저 train 상호작용 중 최근 N=50 아이템 임베딩의 exp-decay(τ=180d) 가중 평균). `use_dynamic` 토글, train split만 사용(누수 없음, 2,001,199 inters=80% 확인). `eval_slices.py`에 `_build_recency` + `IKGR_dyn` spec.

**TO ablation (12ep, 3seed, mean±std) — `run/slice_eval_TO_result.json`:**
| 모델 | overall NDCG@10 | tail Recall@10 | tail@30 | cov@10 |
|---|---|---|---|---|
| MF (kgoff) | 0.0777±.0005 | 0.0089±.0003 | 0.0220 | 0.627 |
| IKGR (full hetero) | 0.0696±.0017 | 0.0108±.0007 | 0.0269 | 0.690 |
| **IKGR+DynLLM(recency)** | 0.0718±.0011 | 0.0110±.0003 | 0.0268 | 0.670 |
| BPR | 0.0778±.0004 | 0.0092±.0004 | 0.0227 | 0.633 |
| LightGCN | 0.0836±.0002 | 0.0052±.0002 | 0.0127 | 0.332 |

- `IKGR → +recency`: overall +3.2%(0.0696→0.0718, KG로 잃은 정확도 일부 회복), tail@10 유지(분산↓), coverage 여전히 높음. → 컴포넌트가 ablation에서 값을 더함(정확도 회복 + long-tail 유지).
- 한계: 이득 modest, overall은 아직 MF/BPR 미만. 단 long-tail/coverage 우위는 유지.
- **다음(Step 3): multi-facet attention fusion** (스칼라 게이트 → MHA). 이후 crowds(Step 4, 선택).


### ▶ DynLLM Step 3: multi-facet attention fusion — 시도 후 기각(negative)
모델: `profile_attn` 토글 — facet 스칼라 게이트 합산 대신 MHA(query=base, key/value=facet 벡터). 점검(체크포인트 α): shelf 2.16 > author 1.97 > recency 1.51 > intent 1.13 > pub 1.10, 모든 게이트 양수(죽은 컴포넌트 없음, recency 활성 확인).

**TO ablation (12ep, 3seed):**
| 모델 | overall NDCG@10 | tail Recall@10 | cov@10 |
|---|---|---|---|
| IKGR full_hetero | 0.0696±.0017 | 0.0108±.0007 | 0.690 |
| IKGR+recency (Step2) | 0.0718±.0011 | 0.0110±.0003 | 0.670 |
| IKGR+recency+attn (Step3) | 0.0727±.0016 | 0.0097±.0002 | 0.642 |

- attention fusion: overall +1.3%이나 **tail −12%, coverage −4%** → facet collapse로 long-tail/다양성(우리 핵심 강점) 손상. ~1.3배 느림. **기각.**
- **결론: DynLLM 실질 기여 = recency(Step 2). attention(Step 3)은 negative.** Step 4(crowds)는 효과 기대 낮아 보류 권장.
- **DynLLM 단계 잠정 종료:** 정직한 스토리 = "recency 동적 프로필이 정확도 일부 회복 + long-tail 유지; 학습 attention fusion은 다양성을 희생해 부적합".


### ▶ CORONA 단계 — Step 1: 3-3 가중합 late-fusion (= Full 모델) → 시도 후 기각(negative)
설계 `CORONA_INTEGRATION.md`(Option B, 레포는 참조만). 다이어그램 3-3 `Final=α·intent+β·dynKG+γ·itemSim`을 채널별 분리 점수 + 학습 가중으로 재구현. `model_ikgr.py`에 `use_corona` 토글(`_corona_user/item_channels`/`_corona_pair`/`_corona_full`), `eval_slices.py`에 `IKGR_full` spec. 채널: CF `<e_u,e_i>` / intent+meta-KG `<kg_u,kg_i>` / recency `<dyn_u,e_i>`, BPR 학습.

**TO ablation (12ep, 3seed, mean±std) — `run/slice_eval_TO_result.json`:**
| 모델 | overall NDCG@10 | tail Recall@10 | tail@30 | cov@10 |
|---|---|---|---|---|
| MF (kgoff) | 0.0777±.0005 | 0.0089 | 0.0220 | 0.627 |
| IKGR (full hetero) | 0.0696±.0017 | 0.0108 | 0.0269 | 0.690 |
| **IKGR+DynLLM (recency)** | **0.0718±.0011** | **0.0110** | **0.0268** | 0.670 |
| Full (+CORONA late-fusion) | 0.0678±.0020 | 0.0101 | 0.0236 | 0.715±.062 |
| BPR | 0.0778±.0004 | 0.0092 | 0.0227 | 0.633 |
| LightGCN | 0.0836±.0002 | 0.0052 | 0.0127 | 0.332 |

- `IKGR+DynLLM → Full`: overall **−5.6%**, tail@10 **−8%**, tail@30 **−12%** (핵심 강점 손상). cov만 +6.7%지만 **분산 ±0.062로 불안정**(seeds 0.628/0.760/0.757).
- **진단:** 학습 가중이 **CF로 붕괴(γ=4.44 ≫ β=2.76 ≫ α=1.20)** → long-tail 견인 KG 채널 억압. + 명시적 분리가 암묵적 내적의 cross-term(`<recency_u,kg_i>` 등)을 버려 표현력↓.
- **판정: 기각(negative).** DynLLM attention fusion과 동일 패턴. 검증된 스토리는 여전히 IKGR+DynLLM 트레이드오프(정확도 일부 양보 ↔ long-tail/coverage robust 우위). 리포트: `CORONA_REPORT.md`.
- **남은 CORONA(보류):** Step 2(3-1 그래프 후보생성, 무료, 별도 메커니즘) / Step 3(3-2 LLM 필터, 유료). 본 무대는 sparse 코어(k=20/30)·글로벌 시간 split.
- 재현: `IKGR_SPLIT=TO IKGR_EPOCHS=12 IKGR_SEEDS=2020,2021,2022 IKGR_SPECS=IKGR_full python eval_slices.py`. 커밋: `65cbf51`(설계+레포), `1bb9a54`(Step1 구현).


### ▶ CORONA Step 2 — 3-1 그래프 후보생성 → 구현·평가 완료, naive 버전은 기각(negative)
`ikgr_core/corona_retriever.py`: train-only·LLM 무관·결정적 retriever. 채널 = intent(user→intent→item) + shelf/author/pub(user 히스토리→meta→item) + CF item-cooccurrence(`Co=UIᵀUI`), scipy.sparse 공기여 합으로 top-M 후보. `eval_slices.py`에 `corona_cand=M` spec knob(후보 외 -inf 마스킹) + `IKGR_cand` spec(=IKGR_dyn 모델, M=500) + 후보 recall@M·크기 기록. 평가 전용(학습 불변), Step1 late-fusion과 독립.

**TO ablation (12ep, 3seed, mean±std):**
| 모델 | overall NDCG@10 | tail Recall@10 | tail@30 | cov@10 | cand recall@500 |
|---|---|---|---|---|---|
| IKGR+DynLLM (full-sort) | 0.0718±.0011 | 0.0110 | 0.0268 | 0.670 | — |
| IKGR_cand (M=500) | 0.0757±.0007 | 0.0035 | 0.0060 | 0.283 | 0.354 |

- 후보제한: overall **+5.4%**(정밀도↑) but tail@10 **−68%**·tail@30 −78%·coverage **−58%** → 핵심 강점 붕괴, 매우 안정(분산~0).
- **진단:** 후보 prior 인기 편향(CF 공기여 + ubiquitous shelf 지배) → tail 배제, 후보 recall 천장 0.354. TO split이라 과거 이웃이 미래 아이템 잘 못 덮음. naive 후보생성 = LightGCN형(overall↑/tail·diversity↓).
- **판정: 기각(negative, 우리 스토리 기준).** mechanism은 완성·재현 가능. **개선 후보:** CF off + intent/meta 가중↑ + popularity 정규화(retriever `weights`/`use_cf` 지원) → 다양성 강화 재시도 시 긍정 여지(미실행).
- 재현: `IKGR_SPLIT=TO IKGR_EPOCHS=12 IKGR_SEEDS=2020,2021,2022 IKGR_SPECS=IKGR_cand python eval_slices.py`. 커밋: `481d8d8`. 리포트: `CORONA_REPORT.md` §5b.


### ▶✅ CORONA Step 2b — 인기 편향 제거 후보생성 → positive (첫 긍정 CORONA 결과)
naive 후보생성(인기 편향)을 제거: `corona_retriever.py`에 **idf**(item↔node 행렬 컬럼 IDF 가중 → "to-read"/"children" 등 ubiquitous 노드 억제) + **pop_norm**(후보점수 ÷ item_pop^β) 추가. `use_cf=False`로 인기 편향 CF 채널 제거. `eval_slices.py`에 `corona_cf/corona_idf/corona_popnorm/corona_weights` 전달 + `IKGR_cand_db` spec(cf off, idf on, pop_norm=0.5).

**TO ablation (12ep, 3seed, mean±std):**
| 모델 | overall NDCG@10 | tail Recall@10 | tail@30 | cov@10 | cand rec@500 |
|---|---|---|---|---|---|
| IKGR+DynLLM (full-sort) | 0.0718 | 0.0110 | 0.0268 | 0.670 | — |
| IKGR_cand (naive) | 0.0757 | 0.0035 | 0.0060 | 0.283 | 0.354 |
| **IKGR_cand_db (편향제거)** | 0.0288 | **0.0275±.0005** | **0.0577** | 0.634 | 0.145 |
| BPR | 0.0778 | 0.0092 | 0.0227 | 0.633 | — |
| LightGCN | 0.0836 | 0.0052 | 0.0127 | 0.332 | — |

- **편향제거 = ✅ positive:** tail Recall@10 0.0275 = **IKGR+DynLLM +150%·BPR +199%·LightGCN +429%** (robust, std 0.0005). tail@30 0.0577 = full-sort +115%·LightGCN +354%. coverage 0.634 ≈ full-sort(다양성 유지).
- **대가:** overall NDCG −60%(0.0718→0.0288). pop 정규화가 인기(미래에도 자주 relevant) 아이템을 후보에서 배제 → 후보 recall 천장 0.145로 더 낮음.
- **해석:** 명시적 diversity-first 검색 = CORONA의 검증된 기여(long-tail 2.5배·diversity 유지, overall 양보). pop_norm/idf가 트레이드오프 노브(덜 공격적 균형점 탐색 가능, 미실행).
- 재현: `IKGR_SPLIT=TO IKGR_EPOCHS=12 IKGR_SEEDS=2020,2021,2022 IKGR_SPECS=IKGR_cand_db python eval_slices.py`. 커밋: `22abbe4`. 리포트: `CORONA_REPORT.md` §5b.
- **CORONA 무료 단계 종료.** 졸업작품 스토리 완성: IKGR(intent+meta KG, long-tail robust) → DynLLM(recency, 정확도 회복+long-tail 유지) → CORONA(편향제거 후보생성, long-tail 2.5배·diversity). 남은 것: Step 3 LLM 필터(유료, 슬라이스 한정) 또는 2순위 k-파라미터 오케스트레이션 → k 스윕.


### ▶✅ k-스윕 2순위: 파이프라인 k-파라미터화 + k=50 실행 (가설 robust 확정)
`run_pipeline.py --k K`로 격리된 per-k 레이아웃(`data/kc_k{K}/`, `run_k{K}/` + `config.k{K}.yaml`)에서 전체 파이프라인(A apply_k_core / B step1 / C step2 / D banks+KG+meta+convert+timestamp / E eval) 실행. config-읽는 스크립트는 `IKGR_CONFIG` env로 redirect. k=100 캐시 재사용으로 비용 절감(프로필 텍스트 키).

**k=50 실행:** 유저 33,070 / 아이템 17,537 / 인터랙션 4,740,194. LLM 신규 호출 24,410건(k=100 캐시 15,198 재사용) ≈ $12~15 실비. meta-KG: author 9,004 / publisher 1,948 / shelf 1,421.

**⚠️ 평가 폭증 버그 수정 (커밋 `78b86e2`):** full_sort_predict가 유저 배치마다 전체 아이템 KG/meta 임베딩을 재계산 → k=100엔 OK였으나 k=50(유저 3배·아이템 2.5배·meta 4관계)에서 spec당 수 시간으로 폭발. **lazy all-item 캐시**(`_ensure_item_cache`, 학습 스텝마다 무효화)로 사후 평가 + RecBole 내부 검증 모두 커버. full_hetero 1ep 113초로 정상화. + 잠복 버그 `def _propagate` 누락 복구.

**k=50 결과 (3-seed, TO, 12ep, mean±std) — `run_k50/slice_eval_TO_result.json`:**
| 모델 | NDCG@10 | tail@10 | tail@30 | cov@10 |
|---|---|---|---|---|
| MF (kgoff) | 0.0735±.0014 | 0.0067 | 0.0160 | 0.452 |
| IKGR (full_hetero) | 0.0688±.0012 | 0.0102 | 0.0222 | 0.600 |
| IKGR+DynLLM | 0.0711±.0017 | 0.0120 | 0.0255 | 0.612 |
| CORONA (cand_db) | 0.0184±.0001 | **0.0246** | **0.0458** | 0.526 |
| BPR | 0.0731±.0010 | 0.0064 | 0.0155 | 0.455 |
| LightGCN | 0.0774±.0003 | 0.0028 | 0.0074 | 0.225 |

**핵심 (가설 robust 확정): sparse해질수록 long-tail 우위 증가 (tail@10 vs MF):**
| 모델 | k=100 | k=50 |
|---|---|---|
| IKGR | +21% | **+52%** |
| IKGR+DynLLM | +24% | **+79%** |
| CORONA | +209% | **+267%** |

- 3시드 모두 분산 ~0으로 robust(frozen+meta-KG는 learnable intent node의 불안정성 없음). 모든 컴포넌트의 long-tail 이득이 k=100→k=50에서 전부 증가 → "intent-KG/통합은 sparse·long-tail에서 빛난다"를 다중시드로 입증.
- CORONA cand_db: diversity-first 트레이드오프(overall 0.018 폭락, tail 압도적 1위). LightGCN: overall 1위·tail/coverage 최악(인기편향 심화).
- 재현: `python run_pipeline.py --k 50` (또는 `--steps E --seeds 2020,2021,2022`). 커밋: `3479af1`(오케스트레이터), `78b86e2`(eval 캐시 수정), `de5de62`(k=50 seed-2020).
- **다음 후보:** k=30 스윕(추세 1점 더, LLM 추가 ~$? 실측 필요, 예산 내) / CORONA Step 3 LLM 필터(슬라이스 한정).


### ▶✅✅ 글로벌 시간 split (cold-start) — 가장 약했던 주장 정면 입증 (핵심 결과)
`IKGR_SPLIT=TO_GLOBAL`(단일 글로벌 시간 컷 70/10/20, group_by=None) + `cold_abs_buckets`(train 인터랙션 수 절대 버킷: 0 / 1-5 / 6-20 / >20). k=50 데이터 재사용(LLM 0). de-risk로 cold 유저 생존 확인(cold0=1215). 커밋 `4d54792`(코드).

**k=50 TO_GLOBAL (3-seed, 12ep) — `run_k50/slice_eval_TO_GLOBAL_result.json`:**

순수 cold 유저(train 0개, 1,215명) Recall@10 / NDCG@10:
| 모델 | cold0 R@10 | cold0 NDCG@10 | cold0 R@30 |
|---|---|---|---|
| MF | 0.0022 | 0.0222 | 0.0058 |
| BPR | 0.0019 | 0.0187 | 0.0053 |
| LightGCN | 0.0056 | 0.0543 | 0.0132 |
| **IKGR** | **0.0485 (~22×MF)** | **0.4350 (~20×MF)** | **0.1089 (~19×MF)** |
| IKGR+Dyn | 0.0458 | 0.4120 | 0.1044 |
| CORONA | 0.0083 | 0.0980 | 0.0129 |

단조 추세 (Recall@10, 유저가 warm해질수록 KG 우위↓): cold0 IKGR/MF +2100% → 1-5 +58% → 6-20 +24% → warm(>20) +2%(동률).

overall NDCG@10 (글로벌 split): **IKGR 0.120 > LightGCN 0.097 > BPR 0.090 ≈ MF 0.090**. cov: MF 0.535 / IKGR 0.366.

- **핵심:** cold 유저는 id-임베딩이 랜덤 → CF(MF/BPR/LightGCN)는 사실상 랜덤 추천(R@10 ~0.002). IKGR은 **유저 프로필 기반 intent/meta-KG로 임베딩을 채워** 이력 0에도 의미 있는 추천(R@10 0.0485). "intent-KG가 cold-start를 해결한다"의 직접 증명. + 글로벌 split에선 IKGR가 overall 정확도도 1위(per-user TO의 정확도 양보가 cold 섞이면 역전).
- CORONA cand_db: cold0 약함(0.0083) — diversity/long-tail 도구지 cold-start 도구 아님(예상대로). IKGR+Dyn: cold0서 IKGR보다 미세 하락(recency는 이력 필요).
- 재현: `python run_pipeline.py --k 50 --steps E --split TO_GLOBAL --seeds 2020,2021,2022 --specs ...`. 무료(k=50 산출물 재사용).
- **졸업작품 약점 #2(cold-start 미검증) 해소.** 이제 스토리: IKGR은 (1) cold-start에서 CF를 압도(~20×), (2) sparse할수록 long-tail 우위↑(k-스윕), (3) CORONA는 diversity-aware 트레이드오프. 정직하고 강한 3단 서사.
