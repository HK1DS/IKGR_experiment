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

**다음 작업 (최우선): cold-start/long-tail 슬라이스 평가.** 유저 인터랙션 수 / 아이템 인기도로 test를 슬라이스해 KG-on vs KG-off vs BPR/LightGCN 비교. 여기서 KG-on이 이기면 그게 졸업작품의 핵심 주장. (이후 α 튜닝, k=20/30 코어 재고, DynLLM/CORONA.)

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
