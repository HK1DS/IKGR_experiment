# IKGR 실험 정리 리포트

졸업작품 3단 통합 프레임워크(IKGR → DynLLM → CORONA)의 **1단계 IKGR**에 대한 구현·실험·결론 정리.
DynLLM 진입 직전까지의 모든 작업을 담는다. (작성 시점: IKGR 단계 종료)

---

## 1. 목표와 결론 요약 (TL;DR)

- **목표:** LLM으로 구매/독서 의도(intent)를 추출해 intent-aware Knowledge Graph를 만들고, 이를 추천에 활용했을 때의 효과를 정직하게 측정한다.
- **핵심 결론:**
  1. 기존 레포의 IKGR 스코어는 **버그 수준**으로 동작하지 않았다 (test NDCG@10 0.045, 인기순 Pop 0.154보다도 낮음). → **재설계로 0.29대까지 복구**(BPR/LightGCN 동급).
  2. intent-KG의 효과를 다중 시드로 검증한 결과, **long-tail "정확도(Recall)" 향상은 robust하지 않았다** (KG-on ≈ KG-off, 분산 큼).
  3. KG의 **일관되게 재현되는 유일한 효과는 추천 "다양성(coverage/novelty) 증가"**이며, 그 대가로 overall 정확도를 소폭 희생한다. → "정확도 vs 다양성 트레이드오프".
  4. **cold-start 주장은 현재 데이터(k=100 dense core)로는 검증 불가** — 가장 비활동 유저도 인터랙션 100+개.

---

## 2. 데이터셋

- **원본:** Goodreads Children (interactions ~1,005만 행, profiles ~37GB).
- **필터:** 표준 **k=100 K-core** 적용 (재현 가능).
  - 유저 **11,073** / 아이템 **6,857** / 인터랙션 **2,489,355** (유저당 평균 ~225 인터랙션 → **dense**).
  - rating 분포: 0이 854,363행(34%, "읽을 예정" 미평가), 4★ 582K, 5★ 563K, 3★ 396K, 2★ 77K, 1★ 16K.
- **intent vocab:** 53,190개 (step1 exact intent 집합), mpnet 임베딩 `intents_emb.npy` [53190, 768].
- 산출물: `data/k_core/interactions_k100.csv`, `profiles_k100.csv`.

---

## 3. 파이프라인 (step1 → step3)

| 단계 | 스크립트 | 내용 | 산출물 |
|---|---|---|---|
| A | `apply_k_core.py` | 원본에 k=100 K-core (메모리 안전 스트리밍) | `data/k_core/*` |
| 1 | `step1.py` | LLM(gpt-4o-mini via Luxia)으로 유저/아이템 의도 추출 (8워커, 캐시) | `run/step1_intents.csv` (11,073행) |
| 2 | `step2.py` | RAG로 관련 의도 확장 (vocab 인코딩 + ANN + LLM 선택) | `run/step2_related_intents.csv` |
| D | `build_intent_banks.py`, `convert_to_recbole_atomic.py` | 임베딩 뱅크 + RecBole atomic(.inter/.kg) | `run/*_bank.pt`, `data/k_core/ikgr-custom/*` |
| KG | `build_kg.py` | intent 노드 + user/item↔intent 인접 패킹 | `run/kg_pack.pt` (intent 53,190 / user edges 275,554 / item edges 174,535) |
| 3 | `step3.py` | IKGR 학습/평가 (RecBole) | `run/ikgr_*_result.json`, 체크포인트 |

### 진행 중 해결한 주요 버그/이슈
- **step2 `p_rel` 미정의** → 프롬프트 로딩 추가.
- **`apply_k_core` OOM** → 청크 스트리밍으로 재작성.
- **LLM 출력 파싱** → ```python 코드펜스 제거 후 리스트 추출(`parse_intent_list`).
- **Windows Annoy 크래시** (`AnnoyIndex.build()` 세그폴트) → `rag.py`가 win32에서 sklearn `NearestNeighbors`로 자동 폴백.
- **step3 스택 오버플로(0xC00000FD)** → KGCN의 노드별 파이썬 루프(18k 깊이 autograd) 제거.
- **8GB VRAM OOM / 느린 평가** → intent cap + 내적 기반 스코어로 재설계(아래).
- **`build_intent_banks` 초저속** → 고유 intent 1회 배치 인코딩으로 ~14초.

---

## 4. 모델 (IKGR) — 진단과 재설계

### 4.1 기존(버그) 모델
스코어: `score = y_ui + λ·z_ui` (λ=0.1)
- `z_ui = cos(user_emb, item_emb)` — 학습되는 부분, ×0.1 → 범위 [-0.1, 0.1]
- `y_ui = max_pair cos(intent_u, intent_i) × penalty(0.5 or 1.0)` — **고정(미학습)**, 범위 ~0.5~1.0

**진단(데이터):** 랜덤 쌍 `y_val` 평균 0.804, 22.5%가 >0.9. penalty cliff(0.9)로 `y_ui`가 bimodal 포화 → 유저당 수백~1000+ 아이템이 top에 동점. 분산 0.257 ≫ λ상한 0.1 → **고정 휴리스틱이 랭킹을 지배, 학습 신호 무력화** → Pop보다 낮은 0.045. 또한 KGCN/`.kg`는 모델에서 실제로 미사용.

### 4.2 재설계 모델 (`ikgr_core/model_ikgr.py`)
- **공유 intent 노드**를 mpnet 임베딩으로 초기화(고정 랜덤 투영, JL 보존). `intent_learnable`로 학습/고정 토글.
- **희소 인접행렬**: user→intent, item→intent (row-normalized 평균).
- **벡터화 전파** (`kg_layers`):
  - L1: `u' = e_u + α·mean(연결된 intent 노드)`, `i'` 동일.
  - L2: intent 노드가 user+item에서도 aggregate → user→intent→item **협업 신호 2-hop** 전파.
- **스코어 = ⟨u', i'⟩ (내적), BPR 학습.** 고정 휴리스틱/penalty 제거.
- `use_kg=False` → 순수 MF/BPR (ablation 기준선).
- 효율: 학습은 배치 intent만 gather(`kg_cap`), 평가는 내적 full-sort(수초). 8GB에서 안전.

---

## 5. 실험 프로토콜

- 프레임워크: RecBole 1.2.0, PyTorch 2.3.0+cu121, GPU **RTX 3060 Ti (8GB)**.
- 공통: 동일 split(RS 0.8/0.1/0.1, group_by user, full-sort), seed=2020(robustness는 2020/2021/2022), embedding_size=512, lr 1e-3, reg 1e-6, uniform neg sampling, eval_step=5.
- 지표: Recall/MRR/NDCG/Hit/Precision @{10,30}, + coverage@10(카탈로그 커버리지), novelty(추천 아이템 self-information bits).
- 슬라이스: 유저 활동도(마스킹 히스토리 수) 분위수 버킷 / long-tail = 인기 하위 80% 아이템.
- **주의:** §6.1/6.2는 20 epochs 단일 시드, §6.3은 12 epochs × 3 시드 → **표 간 직접 비교 금지, 표 내부만 비교.**

---

## 6. 결과

### 6.1 Overall 비교 (20 epochs, seed 2020, test) — "버그 수정 효과"

| 모델 | NDCG@10 | Recall@10 | MRR@10 | Hit@10 | NDCG@30 |
|---|---|---|---|---|---|
| IKGR (구버전, 버그) | 0.045 | 0.021 | 0.125 | 0.296 | 0.049 |
| Pop | 0.1541 | 0.0765 | 0.3626 | 0.6471 | 0.1531 |
| IKGR KG-off (=MF) | 0.2936 | 0.1526 | 0.5652 | 0.8794 | 0.3013 |
| BPR | 0.2953 | 0.1537 | 0.5661 | 0.8819 | 0.3024 |
| LightGCN | 0.2964 | 0.1484 | 0.5631 | 0.8671 | 0.2983 |
| IKGR KG-on | 0.2810 | 0.1480 | 0.5469 | 0.8719 | 0.2903 |

- 버그 수정으로 **0.045 → 0.28~0.29대**. KG-off(MF) 0.2936 ≈ BPR 0.2953 → 구현 정상(sanity check).
- KG-on(0.281) < KG-off(0.294): dense 데이터에서 KG 전파가 overall을 소폭 희석.

### 6.2 슬라이스 평가 (20 epochs, seed 2020, custom per-user eval)

**Long-tail Recall (인기 하위 80% 아이템, ~5,487개):**

| 모델 | tail Recall@10 | tail Recall@30 |
|---|---|---|
| IKGR KG-on | **0.0734** | **0.1415** |
| IKGR KG-off | 0.0704 | 0.1346 |
| BPR | 0.0581 | 0.1160 |
| LightGCN | 0.0450 | 0.0934 |

→ 단일 시드에선 **KG-on > KG-off > BPR > LightGCN** (깨끗한 순서). 그러나 §6.3에서 재현 안 됨(주의).

**Cold-start 버킷 (유저 활동도 분위수):** 활동도 분위수 컷 = **[105, 126, 162, 245]**.
→ 가장 cold한 Q1조차 **인터랙션 105개+** = k=100 코어가 진짜 cold 유저를 제거. **cold-start 검증 불가.** (Q1 NDCG@10: kgoff 0.244 > BPR 0.232 ≈ kgon 0.230 — KG 우위 없음)

### 6.3 다중 시드 robustness (12 epochs, seeds 2020/2021/2022, mean±std) — **가장 신뢰할 결론**

| 모델 | overall NDCG@10 | tail Recall@10 | tail Recall@30 | coverage@10 | novelty |
|---|---|---|---|---|---|
| IKGR KG-off (MF) | 0.2922±.0009 | 0.0597±.0008 | 0.1180±.0010 | 0.6401±.0004 | 10.678±.009 |
| IKGR KG-on L1 (학습노드) | 0.2423±.0030 | 0.0585±**.0104** | 0.1140±**.0155** | 0.7382±**.0874** | 10.969±.194 |
| IKGR KG-on L1 (frozen) | 0.2782±**.0002** | 0.0580±**.0002** | 0.1151±.0004 | 0.6683±.0036 | 10.734±.013 |
| BPR | 0.2935±.0021 | 0.0600±.0015 | 0.1186±.0013 | 0.6431±.0047 | 10.672±.025 |
| LightGCN | 0.2585±.0017 | 0.0276±.0005 | 0.0566±.0006 | 0.3028±.0036 | 9.936±.004 |

(LightGCN은 12 epochs라 overall 저평가; 20ep에선 0.296.)

**해석:**
1. **단일 시드 long-tail 우위(0.0734)는 재현되지 않음.** 3시드 평균 tail Recall@10: KG-on 0.0585 ≈ KG-off 0.0597 ≈ BPR 0.0600 (사실상 동률).
2. **학습 가능 intent 노드(27M 파라미터)는 불안정** (tail std ±0.0104 ≈ ±18%, seed 2021은 0.044로 붕괴). frozen 변형은 분산 ~0.0002로 매우 안정 → instability 원인 = 학습 노드 확정.
3. frozen조차 tail에서 MF를 못 넘음(0.0580 < 0.0597). → **어느 KG 변형도 long-tail 정확도를 robust하게 개선 못 함.**
4. **유일하게 일관된 KG 효과 = 다양성:** coverage@10 KG-on 0.668(frozen)~0.738(학습) vs MF 0.640; LightGCN은 0.303(popularity bias 확정). novelty도 KG-on이 높음. 단 overall 정확도는 희생(0.24~0.28 vs 0.29).

---

## 7. 종합 결론 (논문에 쓸 수 있는 형태)

- intent-KG는 이 dense 데이터에서 **정확도 우위를 제공하지 않는다**(robust하지 않음). 단일 시드만 보면 과대 주장할 위험이 있었고, **다중 시드가 이를 바로잡았다.**
- intent-KG의 **검증된 기여는 추천 다양성/coverage 증가**이며, 이는 **정확도-다양성 트레이드오프**로 정직하게 서술 가능하다.
- 미약한 long-tail 정확도 이득(20ep 단일시드에서 +4%)은 **긴 학습에서만 조건부**로 나타나고 크지 않다.
- 강한 CF(LightGCN)는 overall 최강이지만 long-tail/coverage 최약 → **popularity bias** 현상을 데이터로 확인.

## 8. 한계 & 다음 단계 (IKGR 범위 밖)

- **cold-start 검증:** k=100 dense core로는 불가 → **k=20/30 sparse core** 필요. 단 LLM 비용 폭증($120~150 추정 > 잔여 크레딧 $59) → **Qwen-turbo/DeepSeek 전환** 검토(`config.yaml`+`.env`만 수정).
- **KG 정확도 개선 여지:** α/reg/epochs 안정화, L2 본격 학습(미니배치 최적화 필요), intent 노드 정규화.
- **rating>0 필터** 변형 비교(현재 rating=0 34% 포함).
- **3단 통합:** DynLLM(엣지 가중치 동적 갱신) → CORONA(동적 탐색+LLM 필터+가중합 랭킹). ablation에서 동일 IKGR을 컴포넌트로 재사용.

## 9. 재현 방법

```bash
# 데이터/의도 (LLM 유료)
python apply_k_core.py --profiles_in data/profiles.csv --interactions_in data/interactions.csv --k 100 --out_dir data/k_core
python step1.py
python step2.py
# 뱅크/포맷/KG (무료)
python build_intent_banks.py --step2_csv run/step2_related_intents.csv --user_out run/user_bank.pt --item_out run/item_bank.pt
python convert_to_recbole_atomic.py --interactions data/k_core/interactions_k100.csv --intents run/step2_related_intents.csv --out_dir data/k_core --dataset ikgr-custom
#  → data/k_core/ikgr-custom.{inter,kg} 를 data/k_core/ikgr-custom/ 폴더로 이동
python build_kg.py

# 학습/평가 (무료, GPU)
IKGR_USE_KG=1 python step3.py     # KG-on  → run/ikgr_kgon_result.json
IKGR_USE_KG=0 python step3.py     # KG-off → run/ikgr_kgoff_result.json
python run_baselines.py           # Pop/BPR/LightGCN (동일 split)

# 다중 시드 슬라이스 평가
IKGR_EPOCHS=12 IKGR_SEEDS=2020,2021,2022 \
IKGR_SPECS=IKGR_kgoff,IKGR_kgon_L1,IKGR_kgon_L1_frozen,BPR,LightGCN \
python eval_slices.py             # → run/slice_eval_result.json
```
Windows에서는 `python` 대신 `.venv\Scripts\python.exe` 사용. env 설정은 PowerShell에서 `$env:NAME="값";` 형식.

## 10. 산출물 / 결과 파일

| 파일 | 내용 |
|---|---|
| `run/ikgr_kgon_result.json`, `run/ikgr_kgoff_result.json` | step3 overall (20ep, seed 2020) |
| `run/baselines_result.json` | Pop/BPR/LightGCN overall (20ep) |
| `run/slice_eval_singleseed_result.json` | 단일시드 슬라이스(20ep) — cold-start 버킷 + long-tail |
| `run/slice_eval_result.json` | 다중시드(12ep×3) mean±std + coverage/novelty |
| `run/recbole/*.pth` | 학습 체크포인트 |

## 11. 핵심 코드 변경 (커밋)

- `98b9f07` IKGR 스코어 버그 수정 + 학습 가능 intent-KG 전파 + baselines
- `2376107` cold-start/long-tail 슬라이스 평가 추가
- `02ef6a7` 다중 시드 robustness + 다양성 지표 + IKGR 단계 종료
- (그 외 사용자 커밋: 실험결과/로그 등)

핵심 파일: `ikgr_core/model_ikgr.py`(모델), `ikgr_core/rag.py`(Annoy 폴백), `step3.py`(학습/평가), `build_kg.py`/`run_baselines.py`/`eval_slices.py`(실험 도구).
