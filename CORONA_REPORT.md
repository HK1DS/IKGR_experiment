# CORONA 단계 실험 정리 리포트

졸업작품 3단 프레임워크의 **3단계 CORONA**. 공식 레포(`CORONA-main/`, SIGIR 2025)를 그대로
이식하는 대신, 다이어그램의 CORONA 메커니즘을 **우리 RecBole/IKGR(+DynLLM) 프레임워크 위에
재구현(Option B)**하여 동일 데이터·split·평가로 `IKGR → +DynLLM → +CORONA(Full)` ablation을
일관되게 측정했다. 설계 근거는 `CORONA_INTEGRATION.md`, 선행 단계는 `IKGR_REPORT.md` / `DYNLLM_REPORT.md`.

---

## 1. TL;DR
- **Step 1(3-3 가중합 late-fusion = Full 모델)을 구현·검증.** 무료, 추가 LLM 0, "통합이 개별보다 낫다"는 ablation 주장에 직접 대응하는 부분.
- **핵심 결론(정직한 negative):** 명시적 채널 분리 + 학습 가중(α,β,γ) late-fusion은 기존 **암묵적 결합임베딩 내적(IKGR+DynLLM)을 이기지 못함.** overall −5.6%, long-tail −8%(@10)·−12%(@30)로 **우리 핵심 강점(long-tail)을 오히려 손상**. 유일한 이득(coverage +6.7%)은 분산이 매우 커서(±0.062) robust하지 않음.
- **진단:** 학습된 가중이 **CF 채널로 붕괴(γ=4.44 ≫ β=2.76 ≫ α=1.20)** → long-tail을 견인하는 KG 채널을 down-weight. 또한 명시적 분리는 암묵적 합에 있던 채널 간 cross-term(`<recency_u, kg_i>` 등)을 버림.
- **판정:** DynLLM attention fusion(Step 3)과 동일하게 **기각**. Step 2(그래프 후보생성)·Step 3(LLM 필터)은 별도 메커니즘이므로 보류/후순위.

---

## 2. 핵심 전제 — 다이어그램 ≠ 공식 레포
- **공식 CORONA 레포**: user-item 이분그래프에서 거리 인지(distance-aware) 유저-이웃 retriever를 학습해 compact 서브그래프를 만드는 coarse-to-fine 프레임워크. LLM은 유저 프로필 임베딩(side info)으로만 사용.
- **다이어그램 CORONA**: (3-1) 동적 KG 탐색→후보생성, (3-2) LLM 후보 필터링, (3-3) 가중합 랭킹 `Final = α·IntentMatch + β·DynamicKG + γ·ItemSimilarity`. → 레포가 구현하지 않는 별도 설계.
- DynLLM 때와 동일하게 **레포는 설계 참조**로만 쓰고 다이어그램 사양을 우리 프레임워크에 재구현.

## 3. 방법 — Step 1: 3-3 가중합 late-fusion (`use_corona`)
기존 모델은 facet들을 더한 **결합 임베딩의 단일 내적** `<u_fused, i_fused>`로 점수를 냄(모든 cross-term을 가중치 1로 암묵 혼합). CORONA Step 1은 이를 **채널별 분리 점수 + 학습 가중**으로 일반화:

```
Final(u,i) = γ·⟨e_u, e_i⟩            # CF / item-similarity 채널
           + α·⟨kg_u, kg_i⟩          # intent + meta-KG 채널
           + β·⟨dyn_u, e_i⟩          # DynLLM recency 채널
```
- `kg_u` = 유저 intent 노드 평균, `kg_i` = item의 intent + author/publisher/shelf 노드 평균.
- `dyn_u` = train-only 최근 N=50 아이템 임베딩의 exp-decay 가중 평균(τ=180d).
- α,β,γ는 학습 가능 스칼라(init 1.0). 구현: `_corona_user_channels`/`_corona_item_channels`/`_corona_pair`/`_corona_full`, BPR 학습. `eval_slices.py`에 `IKGR_full` spec.
- ablation 기준: `IKGR_full`(=IKGR+DynLLM+CORONA) vs `IKGR_dyn`(IKGR+DynLLM) vs `IKGR_full_hetero`(IKGR).

## 4. 결과 (시간순 TO split, 12 epochs, seeds 2020/2021/2022, mean±std)

| 모델 | overall NDCG@10 | tail Recall@10 | tail Recall@30 | coverage@10 |
|---|---|---|---|---|
| MF (kgoff) | 0.0777±.0005 | 0.0089±.0003 | 0.0220 | 0.627 |
| BPR | 0.0778±.0004 | 0.0092±.0004 | 0.0227 | 0.633 |
| LightGCN | 0.0836±.0002 | 0.0052±.0002 | 0.0127 | 0.332 |
| IKGR (full hetero) | 0.0696±.0017 | 0.0108±.0007 | 0.0269 | 0.690 |
| **IKGR+DynLLM (recency)** | **0.0718±.0011** | **0.0110±.0003** | **0.0268** | 0.670 |
| **Full (+CORONA late-fusion)** | 0.0678±.0020 | 0.0101±.0007 | 0.0236 | 0.715±.062 |

`IKGR+DynLLM → Full`: overall **−5.6%**(0.0718→0.0678), tail@10 **−8%**(0.0110→0.0101), tail@30 **−12%**(0.0268→0.0236), cov +6.7%(0.670→0.715, **분산 ±0.062로 불안정**: seeds 0.628/0.760/0.757).

학습된 채널 가중(체크포인트 seed 2022): **γ(CF) 4.44 ≫ β(recency) 2.76 ≫ α(intent/KG) 1.20.**

## 5. 해석 / 결론
- **`CORONA_INTEGRATION.md` §6 리스크가 현실화:** "명시적 분리 점수 + 학습 가중"이 암묵적 합산 대비 실익이 없음 — 오히려 손해.
- **원인 1 — 가중 붕괴:** α,β,γ가 자유롭게 학습되자 모델은 overall loss(BPR)를 최소화하려 **CF 채널(γ=4.44)을 압도적으로 키우고 KG 채널(α=1.20)을 억눌렀다.** dense 코어에선 CF가 overall 신호를 지배하므로 합리적 최적화지만, 그 결과 KG가 주던 long-tail/coverage 이득이 희석됨 → tail Recall 하락.
- **원인 2 — cross-term 손실:** 암묵적 내적 `<u_fused,i_fused>`는 `<recency_u, kg_i>` 같은 채널 간 상호작용을 포함하지만, CORONA의 채널별 분리 점수는 같은 채널 쌍만 곱해 이런 신호를 버린다. 즉 명시적 분리가 **더 표현력이 낮은** 점수 함수가 됨.
- **결론:** Full(CORONA late-fusion)은 dense·per-user 시간 split 환경에서 IKGR+DynLLM 대비 **negative**. DynLLM attention fusion과 동일하게 정직히 기각하며, 졸업작품의 검증된 스토리는 여전히 **IKGR+DynLLM = "정확도 일부 양보 ↔ long-tail/coverage robust 우위" 트레이드오프**다.

## 6. 재현성
- from-scratch 독립 학습(이전 런/체크포인트 비의존). late-fusion·채널 집계·recency 모두 LLM 무관 결정적(seed 고정); intent 채널만 저장된 LLM 출력(`step2_related_intents.csv`)에 의존.
- 재현:
  ```
  IKGR_SPLIT=TO IKGR_EPOCHS=12 IKGR_SEEDS=2020,2021,2022 \
  IKGR_SPECS=IKGR_full python eval_slices.py   # -> run/slice_eval_TO_result.json
  ```
  (비교 행 IKGR_kgoff/IKGR_full_hetero/IKGR_dyn/BPR/LightGCN은 DynLLM 단계에서 동일 파일에 적재됨.)

## 7. 한계 & 다음 단계 후보
- **Step 1(가중합)이 negative**라 Full 통합의 "정확도 우위" 주장은 이 환경에선 성립 안 함. 단 이는 **dense k=100 + per-user 시간 split**의 구조적 한계(CF 홈그라운드)와 일치.
- **남은 CORONA 메커니즘(별도, 보류):**
  - Step 2 (3-1 그래프 후보생성, 무료): full-sort 대신 그래프 이웃 후보군 → cross-category 다양성. 평가 공정성(누수) 설계 주의. coverage 강화 가능성.
  - Step 3 (3-2 LLM 필터, 유료): 소규모 cold/long-tail 슬라이스에서만 검증 권장. Qwen 등 저렴 provider 전제.
- **본 무대:** sparse 코어(k=20/30)·글로벌 시간 split에서 KG/통합이 빛날 가능성(LLM 비용 → Qwen/DeepSeek 전환). 현재 dense 환경에선 통합의 가치는 "정확도"가 아니라 "다양성/coverage"에 한정됨.

## 8. 산출물 / 커밋
- 코드: `ikgr_core/model_ikgr.py`(`use_corona` late-fusion), `eval_slices.py`(`IKGR_full` spec).
- 결과: `run/slice_eval_TO_result.json`(`IKGR_full`). 설계: `CORONA_INTEGRATION.md`. 참조: `CORONA-main/`(공식).
- 커밋: `65cbf51`(설계+레포), `1bb9a54`(Step1 late-fusion 구현).
