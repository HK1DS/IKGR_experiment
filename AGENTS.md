# IKGR Pipeline Progress & Execution Guide

이 문서는 IKGR 파이프라인의 진행 상황 및 새 세션/새 환경에서 이어서 진행할 때 참고하는 실행 가이드입니다.
**(최종 갱신: Step A(k-core 생성) + Step B(step1 의도 추출) 완료, Step C(step2 RAG 확장) 진행 대기 시점)**

### ✅ 현재 진행 현황 한눈에 보기
| 단계 | 내용 | 상태 |
|------|------|------|
| Step A | k=100 K-core 실데이터 생성 | ✅ 완료 (`data/k_core/interactions_k100.csv`, `profiles_k100.csv`) |
| Step B | step1 — LLM 의도 추출 | ✅ 완료 (`run/step1_intents.csv`, **11,073행 전부 채워짐**, exact intent 빈 셀 0) |
| Step C | step2 — RAG 의도 확장 | ⏳ **다음 실행 대상** (유료) |
| Step D | 임베딩 뱅크 + RecBole 포맷 변환 | ⬜ 대기 |
| Step E | step3 — IKGR GNN 학습/평가 | ⬜ 대기 |

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

### Step C. Step 2 — RAG 의도 확장 (⚠️ 유료) — ⏳ **다음 실행 대상**
```bash
python step2.py
```
* **사전 조건 모두 충족 확인됨** (step1 산출물 완전, `prompts/step2_related.txt` 존재 → `p_rel` 정상 로딩, config 경로 정상). 바로 실행 가능.
* 입력 `run/step1_intents.csv` → 출력 `run/step2_related_intents.csv`. step1 exact intent로 vocab 새로 구성 후 RAG 후보 → LLM 선택.
* 약 15K 호출. 캐시(`run/step2_cache.json`)로 중단/재개 가능. **이 단계 들어가기 전 사용자 재확인 필요.**
* **step1과 동일하게 동시성(8 워커) + 원자적 캐시 저장 적용됨.** 단, 인코더(SentenceTransformer) thread-safety 문제 때문에 임베딩은 동시 단계 전에 배치로 미리 계산하고, 스레드에서는 ANN 검색(읽기) + LLM 호출만 수행. 워커 수는 `IKGR_STEP2_WORKERS`로 조절.

### Step D. 임베딩 뱅크 + RecBole 포맷 변환
```bash
python build_intent_banks.py --step2_csv run/step2_related_intents.csv --encoder sentence-transformers/all-mpnet-base-v2 --user_out run/user_bank.pt --item_out run/item_bank.pt
python convert_to_recbole_atomic.py --interactions data/k_core/interactions_k100.csv --intents run/step2_related_intents.csv --out_dir data/k_core --dataset ikgr-custom
```
* 변환 후 RecBole 폴더 정렬 필요: `data/k_core/ikgr-custom.{inter,kg}` → `data/k_core/ikgr-custom/` 폴더로 이동.
  (step3가 `data_path = dirname(inter_file) = data/k_core` 에서 `data/k_core/ikgr-custom/ikgr-custom.inter`를 찾음)

### Step E. Step 3 — IKGR GNN 학습/평가
```bash
python step3.py
```
* IKGR baseline 성능 확보. 이후 DynLLM/CORONA 구현 단계로.

---

## 4. 환경 메모
* Windows + cmd 셸. `python`이 직접 안 잡히면 `.venv\Scripts\python.exe` 사용.
* PowerShell에서 `$_` 등 변수 이스케이프가 깨지는 경우가 있으니, 파일 조작은 짧은 Python 스크립트로 처리하는 게 안전.
* `.gitignore`가 `data/`, `run/` 전체를 제외 → 대용량 데이터/산출물은 깃 추적 대상 아님.
* **인코딩 주의**: Windows 기본 인코딩이 `cp949`라 `run/*.json`(예: step1/step2 캐시)을 점검용으로 직접 열 때 `open(path)`만 쓰면 `UnicodeDecodeError`가 난다. 반드시 `open(path, encoding="utf-8")`로 열 것. (파이프라인 코드 자체는 이미 utf-8로 읽고/쓰므로 정상 동작에는 영향 없음 — 디버깅 스크립트에서만 주의.)
