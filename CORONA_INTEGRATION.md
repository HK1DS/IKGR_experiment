# CORONA 통합 설계 (Option B: 메커니즘 재구현)

졸업작품 3단 프레임워크의 **3단계 CORONA(KG-based / cross-category recommendation)**.
공식 레포(`CORONA-main/`, SIGIR 2025)를 그대로 이식하지 않고, **다이어그램의 CORONA 메커니즘을
우리 RecBole/IKGR(+DynLLM) 프레임워크 위에 재구현**한다. IKGR/DynLLM 단계와 동일한 데이터·split·평가로
`IKGR → +DynLLM → +CORONA(Full)` ablation을 일관되게 측정하는 것이 목적.
선행: `IKGR_REPORT.md`, `DYNLLM_REPORT.md`, `DYNLLM_INTEGRATION.md`.

---

## 0. 핵심 전제 — 다이어그램 ≠ 공식 레포
- **공식 CORONA 레포**는 user-item 이분그래프에서 **거리 인지(distance-aware) 유저-이웃 retriever**를 학습해
  compact 서브그래프를 만드는 coarse-to-fine 프레임워크다. LLM은 유저 프로필 임베딩(side info)으로만 사용.
- **다이어그램의 CORONA**는 (3-1) 동적 KG 탐색→후보 생성, (3-2) LLM 후보 필터링, (3-3) 가중합 랭킹
  `Final = α·IntentMatch + β·DynamicKG + γ·ItemSimilarity` 으로, **레포가 구현하지 않는** 별도 설계다.
- 따라서 DynLLM 때와 같이 **레포는 설계 참조**로만 쓰고, 다이어그램 사양을 우리 프레임워크에 재구현한다.

## 1. 다이어그램 CORONA → 우리 프레임워크 매핑

| 다이어그램 | 공식 레포 대응 | 우리 재구현 | 비용 |
|---|---|---|---|
| 3-1 동적 KG 탐색 → 후보 생성 | 거리 인지 이웃 retrieval | KG/이분그래프 이웃 기반 **후보군 생성**(유저의 intent/meta/행동 이웃 아이템) | 0 |
| 3-2 LLM 기반 후보 필터링 | (없음) | LLM로 후보 prune (추론 시 호출) | ⚠️ 유료·추론지연 |
| 3-3 가중합 랭킹 (α·intent+β·dynKG+γ·itemSim) | (없음) | **IKGR(intent-match) + DynLLM(recency) + item similarity 점수 late-fusion** | 0 |

**핵심 통찰:** 3-3 가중합 랭킹은 사실상 **IKGR+DynLLM 신호를 결합하는 "통합" 그 자체** = 졸업작품의 `Full(1+2+3)` 모델.
가장 싸고, 추가 LLM 0이며, "통합이 개별보다 낫다"는 ablation 주장에 정확히 부합한다. → **최우선.**

## 2. 우선순위 (가치/비용 순)

### Step 1 (최우선, 무료) — 3-3 가중합 late-fusion = Full 모델
- 후보(또는 전체 아이템)에 대해 세 점수를 결합:
  - `s_intent` = IKGR intent/meta-KG facet 기반 유저-아이템 점수 (이미 보유)
  - `s_dyn`    = DynLLM recency 동적 프로필 기반 점수 (이미 보유)
  - `s_sim`    = 기본 임베딩 내적/코사인 (item similarity, CF 신호)
  - `Final = α·s_intent + β·s_dyn + γ·s_sim`
- 구현: 우리 모델은 이미 `_emb_users`(intent+recency)·`_emb_items`(intent+meta)로 **결합 임베딩의 내적**으로 이 합을 암묵적으로 계산 중. CORONA 단계는 이를 **명시적 분리-점수 + 학습/튜닝 가능한 α,β,γ 가중**으로 일반화(late fusion, 게이트 학습 또는 grid).
- ablation: `Full` = 세 신호 결합 vs 각 신호 단독/부분.
- 평가: 기존 TO split + slice(long-tail/coverage)로 `IKGR` / `IKGR+DynLLM` / `Full` 비교.

### Step 2 (중간, 무료) — 3-1 후보 생성 (coarse retrieval)
- full-sort 대신 **유저별 후보군**을 그래프 이웃에서 생성: 유저의 intent/shelf/author 공유 아이템 + recency 이웃 + CF 이웃.
- CORONA의 "거리 prior로 이웃 가중" 아이디어 차용(1-hop/2-hop 가중).
- 효과: cross-category 후보 다양성↑(우리 long-tail 강점과 결합), full-sort 비용↓.
- 단 RecBole full-sort 평가와 정합 위해, 후보군을 랭킹에 쓰되 평가는 동일 프로토콜 유지(후보 외 아이템 -inf 마스킹 등) — 설계 시 누수/공정성 주의.

### Step 3 (선택/후순위, 유료) — 3-2 LLM 필터링
- 후보 top-M을 LLM로 prune/re-score. 추론 시 유저당 LLM 호출 → 비용·지연 큼.
- 권장: 전체 평가셋 대신 **소규모 슬라이스(cold/long-tail 유저)에서만** 검증하거나 후순위. Qwen 등 저렴 provider 전제.

## 3. 의존성 / 환경
- CORONA `requirements-min`은 **torch==1.13.1 핀 + PyG(torch-scatter/sparse 컴파일)** → 우리 torch 2.3과 **충돌**.
- 그러나 우리는 레포를 실행하지 않고 **메커니즘만 재구현**하므로 그 의존성 불필요.
  - 3-3/3-1은 순수 torch(우리 모델 확장)로 구현 → **추가 의존성 0**.
  - 그래프 이웃은 scipy.sparse(이미 보유)로 충분, **PyG 불필요**. (정 필요해도 torch 2.3엔 최신 PyG만, 옛 핀 금지.)
- 3-2 LLM 필터만 LLM 클라이언트 필요(이미 `ikgr_core/llm_client.py` 보유).

## 4. 평가 / ablation 설계
- 동일 TO split·seed·12ep·3seed. 지표: NDCG/Recall@{10,30} + long-tail Recall + coverage + novelty.
- ablation 표(목표):
  | 구성 | 설명 |
  |---|---|
  | MF / BPR / LightGCN | 외부 baseline |
  | IKGR | intent+meta KG |
  | IKGR+DynLLM | + recency |
  | **Full (IKGR+DynLLM+CORONA)** | + 가중합 late-fusion (+선택적 후보생성/LLM필터) |
- 핵심 주장: **Full > 부분 조합**, 특히 long-tail/coverage(우리 검증된 강점)에서.

## 5. 구현 단계 (제안)
1. 모델에 **분리-점수 + α,β,γ 가중 late-fusion** 추가(`use_corona` 토글, 학습형 게이트). → `eval_slices.py`에 `Full` spec.
2. (선택) 그래프 후보 생성기(scipy.sparse 이웃) + 후보 마스킹 평가.
3. (선택) LLM 필터 — 소규모 슬라이스 검증.
4. ablation 재실행 + `CORONA_REPORT.md`.

## 6. 리스크 / 메모
- 3-3가 이미 우리 결합 임베딩 내적과 수학적으로 겹칠 수 있음 → **"명시적 분리 점수 + 학습 가중"이 암묵적 합산 대비 실익이 있는지**가 관건(없으면 negative로 정직히 기록, DynLLM attention 사례처럼).
- 후보 생성(3-1)은 평가 공정성(누수) 설계 주의.
- LLM 필터(3-2)는 비용/재현성 이슈 → 핵심 주장에서 분리.
- gain은 dense k=100에서 modest할 수 있음 → sparse 코어/글로벌 시간 split이 본 무대(별도 과제).

## 7. 산출물(예정)
- 확장된 `ikgr_core/model_ikgr.py`(`use_corona` late-fusion), `eval_slices.py`(`Full` spec[, 후보생성]),
  결과 `run/slice_eval_TO_result.json`, `CORONA_REPORT.md`. 참조: `CORONA-main/`(공식).
