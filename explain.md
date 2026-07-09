# 현재 진행 상황과 Goodreads 실험 종료 판정

최종 확인일: 2026-07-09

## 결론

**졸업작품의 핵심 가설을 Goodreads Children 데이터셋으로 검증하는 실험은 완료됐다고 봐도 된다.**

정확히 말하면 세상에 가능한 모든 추가 실험을 소진한 것은 아니지만, 현재 연구 질문에 필요한 주요 실험은 모두 끝났다. IKGR, DynLLM, CORONA의 구현 및 ablation, 다중 시드 평가, 희소성 k-sweep, 시간순 평가, 글로벌 시간 분할 기반 cold-start 평가까지 완료됐다. 따라서 지금부터는 새로운 장시간 GPU 실험보다 **결과 통합, 표/그림 작성, 보고서 및 발표 자료 정리**가 우선이다.

## 완료된 파이프라인

| 구분 | 완료 내용 | 상태 |
|---|---|---|
| 데이터 | Goodreads Children k-core 생성 | k=100, 50, 30 완료 |
| IKGR 입력 | LLM intent 추출, RAG 확장, 임베딩 뱅크 | 완료 |
| IKGR | intent KG + 저자/출판사/shelf 메타 KG | 완료 |
| DynLLM | recency 기반 동적 사용자 프로필 | 완료 |
| CORONA | late fusion, 그래프 후보 생성, 인기 편향 제거 후보 생성 | 완료 |
| Baseline | MF, BPR, LightGCN | 완료 |
| 평가 | 3 seeds, TO split, long-tail, coverage | 완료 |
| Cold-start | TO_GLOBAL split, train 이력 0명 평가 | k=50, k=30 완료 |
| 희소성 검증 | k=100 → 50 → 30 비교 | 완료 |

## 현재 확보된 핵심 결과

### 1. IKGR: cold-start에서 가장 강한 결과

k=30 TO_GLOBAL에서 순수 cold-start 사용자(train interaction 0명)는 2,736명이다.

| 모델 | cold0 Recall@10 | cold0 NDCG@10 |
|---|---:|---:|
| MF | 0.0022 | 0.0140 |
| BPR | 0.0029 | 0.0171 |
| LightGCN | 0.0078 | 0.0475 |
| **IKGR** | **0.0677** | **0.3867** |

IKGR의 cold0 Recall@10은 MF 대비 약 31배, BPR 대비 약 23배, LightGCN 대비 약 8.7배다. 사용자 이력이 없어 ID 기반 CF가 제대로 학습되지 못할 때, 프로필의 intent와 메타데이터 KG가 실제로 역할을 한다는 핵심 주장이 검증됐다.

### 2. 희소할수록 long-tail 효과가 커짐

k를 100에서 50, 30으로 낮춰 데이터가 희소해질수록 IKGR 계열의 long-tail 이점이 커지는 경향이 확인됐다. k=30 TO의 tail Recall@10은 다음과 같다.

| 모델 | tail Recall@10 | MF 대비 |
|---|---:|---:|
| MF | 0.0105 | 기준 |
| IKGR | 0.0139 | +32% |
| IKGR+DynLLM | 0.0170 | +62% |
| CORONA cand_db | 0.0273 | +160% |

이는 “intent/KG 기반 추천은 dense한 환경의 전체 정확도 경쟁보다 sparse, cold-start, long-tail 환경에서 가치가 크다”는 연구 방향을 지지한다.

### 3. DynLLM: 정확도 회복과 long-tail 유지

recency 기반 동적 프로필은 IKGR의 long-tail 성능을 유지하면서 overall 정확도를 일부 회복했다. k=30 TO에서는 IKGR+DynLLM의 NDCG@10이 0.0822로 MF 0.0795와 IKGR 0.0771보다 높았고, tail Recall@10도 0.0170으로 IKGR 0.0139보다 높았다.

반면 multi-head attention fusion은 tail 및 coverage를 떨어뜨려 기각했다. 이 negative result도 ablation 근거로 보존한다.

### 4. CORONA: diversity-first 트레이드오프

인기 편향을 제거한 CORONA 후보 생성은 long-tail 성능을 크게 높이지만 overall NDCG를 희생한다. k=30 TO_GLOBAL에서 tail Recall@10은 0.0148로 MF의 약 7배, LightGCN의 약 16배지만 overall NDCG@10은 0.0232다.

따라서 CORONA는 전체 정확도 최적화 모델이 아니라 **long-tail/diversity를 우선하는 명시적 제어 단계**로 해석하는 것이 맞다.

## Goodreads 실험은 정말 다 끝났는가?

### 필수 실험: 완료

- IKGR, DynLLM, CORONA 각 컴포넌트 구현 및 ablation
- MF/BPR/LightGCN과 동일 조건 비교
- 3개 seed를 사용한 반복 평가
- 랜덤 분할과 시간순 TO 평가
- k=100/50/30 희소성 sweep
- long-tail 및 coverage 평가
- 글로벌 시간 분할의 순수 cold-start 평가
- 실패한 변형까지 포함한 원인 분석

이 범위면 졸업작품의 핵심 주장과 컴포넌트별 기여를 설명할 근거가 충분하다. **Goodreads에서 필수로 더 돌려야 할 파이프라인은 없다.**

### 선택적으로 남은 실험

아래는 가능하지만 현재 결론을 위해 필수는 아니다.

1. CORONA Step 3 LLM 필터링: 유료이며 결과 개선이 보장되지 않는다.
2. CORONA `pop_norm`, 후보 수 M 등의 세부 튜닝: 정확도와 long-tail 사이의 중간점을 찾을 수 있으나 추가 탐색 비용이 든다.
3. `rating > 0`만 사용하는 데이터 정제 ablation: 현재는 Goodreads의 `to-read` 성격인 rating 0도 positive interaction으로 포함한다.
4. 더 많은 seed 또는 epoch/hyperparameter sweep: 신뢰구간을 더 촘촘히 할 수 있지만 핵심 경향은 이미 3 seeds에서 확인됐다.
5. 다른 데이터셋 재현: Goodreads 내부 검증이 아니라 외적 타당성을 강화하는 후속 연구에 해당한다.

## 권장 다음 단계

1. k=100/50/30 결과를 하나의 최종 비교표로 통합한다.
2. cold-start, long-tail, overall/coverage 트레이드오프 그림을 만든다.
3. positive result와 negative ablation을 구분해 논문/보고서 문장으로 정리한다.
4. `AGENTS.md`, 이 문서, k=30 결과 JSON을 정리해 커밋한다.

현재 상태의 가장 정직한 최종 메시지는 다음과 같다.

> IKGR은 프로필 기반 intent와 이종 KG를 통해 cold-start에서 CF를 크게 앞서고, 데이터가 희소해질수록 long-tail 이점이 커진다. DynLLM의 recency 프로필은 정확도를 보완하며, CORONA의 편향 제거 후보 생성은 overall 정확도를 희생하는 대신 long-tail 다양성을 명시적으로 강화한다.
