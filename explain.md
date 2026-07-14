# 현재 진행 상황과 최종 정리

최종 갱신: 2026-07-14

## 한 줄 결론

Goodreads Children 기준으로 필수 실험은 완료됐다. 현재 결과는 "전체 추천 정확도 SOTA" 주장이 아니라, **LLM intent와 KG가 cold-start/long-tail 상황에서 기존 CF 모델의 약점을 보완한다**는 주장으로 정리하는 것이 가장 방어 가능하다.

## 최종 주장

이 파이프라인의 강점은 모든 상황에서 MF/BPR/LightGCN을 이기는 것이 아니다. 더 정확한 주장은 다음과 같다.

1. **IKGR**: 유저 이력이 없거나 적은 cold-start 상황에서 프로필 기반 intent와 meta-KG를 활용해 CF baseline보다 강하다.
2. **DynLLM**: 현재 구현에서는 LLM-heavy 프로필 생성 전체가 아니라 recency 기반 동적 유저 표현이며, IKGR의 정확도와 long-tail 성능을 일부 보완한다.
3. **CORONA**: 현재 구현에서는 overall accuracy 향상 모듈이라기보다 long-tail/diversity를 명시적으로 강화하는 trade-off 모듈이다.

따라서 최종 메시지는 다음이 적절하다.

> IKGR은 프로필 기반 intent와 이종 KG를 통해 cold-start에서 CF 모델을 크게 앞서고, 데이터가 희소해질수록 long-tail 이점이 커진다. DynLLM의 recency 프로필은 정확도를 일부 보완하며, CORONA는 overall accuracy를 희생하는 대신 long-tail/diversity를 제어하는 모듈로 작동한다.

## 완료된 실험 범위

| 구분 | 상태 |
|---|---|
| Goodreads k-core | k=100, k=50, k=30 완료 |
| LLM intent 추출 | step1 완료 |
| RAG intent 확장 | step2 완료 |
| IKGR | intent-KG, meta-KG, KG-off ablation 완료 |
| DynLLM | recency dynamic profile 완료, attention fusion은 negative |
| CORONA | late-fusion negative, hard candidate, debiased candidate, soft-rerank 완료 |
| Baseline | MF, BPR, LightGCN 완료 |
| 평가 | 3 seeds, TO split, TO_GLOBAL split, cold-start, long-tail, coverage 완료 |
| 재현성 | deterministic rerank 검증 완료 |

## 핵심 결과

### 1. Cold-start: IKGR의 가장 강한 결과

k=30 TO_GLOBAL에서 train interaction이 0개인 순수 cold-start 유저는 2,736명이다.

| 모델 | cold0 Recall@10 | cold0 NDCG@10 |
|---|---:|---:|
| MF | 0.0022 | 0.0140 |
| BPR | 0.0029 | 0.0171 |
| LightGCN | 0.0078 | 0.0475 |
| IKGR | 0.0677 | 0.3867 |

MF/BPR/LightGCN은 유저 이력이 없으면 ID embedding을 제대로 학습할 수 없다. 반면 IKGR은 Goodreads profile에서 추출한 intent와 item metadata KG를 사용하기 때문에 cold-start 유저에게도 의미 있는 추천을 만들 수 있다.

### 2. Sparse/long-tail: 데이터가 희소할수록 KG 계열이 유리

k=30 TO split의 tail Recall@10은 다음과 같다.

| 모델 | tail Recall@10 | MF 대비 |
|---|---:|---:|
| MF | 0.0105 | 기준 |
| IKGR | 0.0139 | +32% |
| IKGR+DynLLM | 0.0170 | +62% |
| CORONA cand_db | 0.0273 | +160% |

이는 intent/KG 기반 추천이 dense한 전체 정확도 경쟁보다 sparse/long-tail 상황에서 더 의미 있다는 점을 보여준다.

### 3. DynLLM: recency는 도움이 되지만, LLM-heavy 구현은 아님

현재 DynLLM 구현은 recency 기반 동적 유저 표현이다. 원 논문 또는 초기 설계처럼 리뷰, 과거 이력, intent, explicit LLM profile generation을 모두 합친 완성형은 아니다.

k=30 TO에서 `IKGR+DynLLM`은 `IKGR`보다 overall NDCG와 tail Recall을 모두 개선했다. 다만 개선폭은 크지 않다. multi-head attention fusion은 tail/coverage를 떨어뜨려 기각했다.

정리하면 DynLLM의 현재 기여는 "LLM 프로필 생성으로 큰 성능 향상"이 아니라, **recency-aware dynamic profile이 IKGR의 약점을 일부 보완한다**는 정도다.

### 4. CORONA: accuracy 모듈이 아니라 diversity trade-off

`CORONA cand_db`는 long-tail 성능을 크게 올리지만 overall NDCG를 크게 낮춘다. k=30 TO_GLOBAL에서 tail Recall@10은 MF 대비 약 7배, LightGCN 대비 약 16배지만 overall NDCG@10은 낮다.

soft-rerank도 추가로 검증했다. deterministic 설정을 고정한 뒤 low-lambda grid를 돌렸지만, 유의미한 accuracy-aware 개선은 없었다.

| lambda | overall NDCG@10 | tail Recall@10 | cold0 Recall@10 | cov@10 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.1150 | 0.0038 | 0.0633 | 0.3477 |
| 0.01 | 0.1150 | 0.0038 | 0.0633 | 0.3481 |
| 0.03 | 0.1149 | 0.0038 | 0.0633 | 0.3491 |
| 0.05 | 0.1147 | 0.0039 | 0.0633 | 0.3503 |

판정: soft-rerank는 재현 가능하게 검증됐지만, 성능 개선으로 채택하기 어렵다. CORONA는 현재 구조에서는 diversity/long-tail trade-off 모듈로 정리하는 것이 맞다.

## MF/BPR/LightGCN 수치가 낮아 보이는 이유

현재 주요 평가는 시간순 split과 global temporal split이다. 특히 TO_GLOBAL은 미래 시점에 등장하는 유저를 test에 포함하므로 cold-start가 강하게 섞인다. 아이템 수가 많고 테스트 조건이 어렵기 때문에 Recall/NDCG 절대값은 낮게 보인다.

따라서 중요한 것은 절대값보다 같은 split, 같은 seed, 같은 데이터에서의 상대 비교다. MF/BPR/LightGCN은 추천 시스템에서 가장 기본적인 CF baseline이므로 선택했다. 이 baseline들과 비교해야 "우리 구조가 어떤 조건에서 유리한지"를 설명할 수 있다.

## 현재 파이프라인의 한계

1. SOTA overall accuracy를 주장하기 어렵다.
2. DynLLM은 비용과 구현 안정성 때문에 recency 중심으로 단순화되어 있다.
3. CORONA는 현재 accuracy 향상보다 long-tail/diversity 쪽으로 작동한다.
4. Goodreads 하나만으로 외적 타당성을 주장하기는 어렵다.
5. soft-rerank까지 확인했지만 Full 모델이 모든 지표를 동시에 개선하지는 못했다.

## 다음에 할 일

현재 Goodreads에서 더 긴 GPU 실험을 계속하기보다 다음 순서가 낫다.

1. 이 결론을 기준으로 보고서/발표 표를 정리한다.
2. IKGR, DynLLM, CORONA 각각의 positive/negative result를 분리해서 설명한다.
3. 다른 데이터셋으로 외적 검증을 준비한다.
4. 다른 데이터셋에서도 같은 프레임을 유지한다: overall accuracy만 보지 말고 cold-start, long-tail, coverage를 함께 본다.

Goodreads 내부 실험은 "필수 실험 완료"로 보는 것이 맞다. 추가 실험은 결과를 뒤집기 위한 작업이 아니라, 다른 데이터셋에서 같은 경향이 재현되는지 확인하는 후속 검증이다.
