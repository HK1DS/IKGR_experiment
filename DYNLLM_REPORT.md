# DynLLM 단계 실험 정리 리포트

졸업작품 3단 프레임워크의 **2단계 DynLLM**. 공식 레포(`dynmLLM/`)를 그대로 이식하는 대신
그 핵심 아이디어를 **우리 RecBole/IKGR 프레임워크 위에 재구현(Option B)**하여,
동일 데이터·split·평가로 `IKGR vs IKGR+DynLLM` ablation을 일관되게 측정했다.
설계 근거는 `DYNLLM_INTEGRATION.md`, IKGR 단계는 `IKGR_REPORT.md` 참고.

---

## 1. TL;DR
- **Option B 채택 이유:** 공식 DynLLM은 TGN 기반 시간 그래프 + 자체 harness라 IKGR(RecBole)과 같은 ablation 표에 못 넣음. 또 코드 미완성(`ProfilesHistoryUpdater` 미정의)·데이터/임베딩 미동봉·4-facet 1536d 임베딩 신규 생성 필요. 의존성(torch/numpy/pandas/sklearn)은 IKGR의 부분집합(충돌 없음).
- **핵심 결론:**
  1. **시간순(TO) 평가로 전환** → IKGR의 long-tail 우위가 RS보다 더 큼(tail@10 +21% vs MF).
  2. **recency 동적 프로필(Step 2) = DynLLM의 실질 기여.** IKGR 대비 overall +3.2% 회복 + long-tail 유지(분산↓).
  3. **multi-facet attention fusion(Step 3) = 기각(negative).** overall만 +1.3%, long-tail −12%·coverage −4%로 핵심 강점 손상.
  4. crowds(Step 4)는 기대효과 낮아 보류.

---

## 2. 데이터 — timestamp 복원 (무료)
- 기존 `interactions_k100.csv`엔 timestamp 없음. 원본 `goodreads_interactions_children.json.gz`의 `read_at/date_added` 복원.
- `goodreads_preprocess.py`(timestamp 인지화) + `add_timestamps.py`(기존 k-core backfill): **2,489,355행 전부 매칭, 99.96% 유효 timestamp** → `interactions_k100_ts.csv`.
- `make_temporal_inter.py`: garbage 날짜(미래 2560/1970 등 ~1.6%)를 [2006,2017]로 clip → `ikgr-custom.inter`에 `timestamp:float` 추가(RS 런 하위호환).

## 3. 방법 (3 스텝)
- **Step 1 — 시간순 split:** `eval_slices.py IKGR_SPLIT=TO` (RecBole `order=TO`, per-user 시간순 80/10/10). IKGR/MF/BPR/LightGCN 재baseline.
- **Step 2 — recency 동적 프로필:** `_emb_users`에 동적 항 추가 = 유저의 **train-only** 최근 N=50 상호작용 아이템 임베딩의 exp-decay(τ=180d) 가중 평균. `use_dynamic` 토글. 누수 없음(train split 2,001,199 inters=80% 확인).
- **Step 3 — attention fusion:** facet 스칼라 게이트 합산 대신 MHA(query=base, key/value=facets). `profile_attn` 토글.
- 평가: NDCG/Recall@{10,30} + long-tail Recall(인기 하위 80% 아이템) + coverage@10 + novelty, 3 시드 mean±std.

## 4. 결과 (시간순 TO split, 12 epochs, seeds 2020/2021/2022, mean±std)

| 모델 | overall NDCG@10 | tail Recall@10 | tail Recall@30 | coverage@10 |
|---|---|---|---|---|
| MF (kgoff) | 0.0777±.0005 | 0.0089±.0003 | 0.0220 | 0.627 |
| BPR | 0.0778±.0004 | 0.0092±.0004 | 0.0227 | 0.633 |
| LightGCN | 0.0836±.0002 | 0.0052±.0002 | 0.0127 | 0.332 |
| IKGR (full hetero) | 0.0696±.0017 | 0.0108±.0007 | 0.0269 | 0.690 |
| **IKGR+DynLLM (recency)** | **0.0718±.0011** | **0.0110±.0003** | 0.0268 | 0.670 |
| IKGR+recency+attn (기각) | 0.0727±.0016 | 0.0097±.0002 | — | 0.642 |

학습된 facet 게이트(체크포인트): **shelf 2.16 > author 1.97 > recency 1.51 > intent 1.13 > pub 1.10** (모두 양수, recency 활성).

## 5. 해석 / 결론
- **시간순 평가는 RS보다 훨씬 어려움**(overall ~0.08 vs ~0.29) — 과거→미래 예측이라 정상. LightGCN이 overall 1위지만 **tail/coverage 최악**(popularity bias가 시간순에서 심화).
- **recency(Step 2)는 ablation에서 값을 더함:** `IKGR → +recency`에서 overall +3.2%(KG로 잃은 정확도 일부 회복) + long-tail 유지(tail@10 0.0110, 분산 절반↓). = "정확도 회복 + 강점 유지".
- **attention fusion(Step 3)은 기각:** overall만 미세 상승, long-tail −12%·coverage −4% (facet collapse). 우리 프레임워크의 핵심 가치(long-tail/다양성)를 훼손하므로 부적합. ~1.3배 느림. → 정직한 negative 결과.
- 종합: **CF(BPR/LightGCN)는 overall 강세지만 long-tail/coverage 약세**, **IKGR+DynLLM은 overall을 일부 양보하는 대신 long-tail Recall(+24% vs MF, +112% vs LightGCN)과 coverage(0.67 vs 0.33)에서 robust 우위.** "정확도 vs long-tail/다양성" 트레이드오프 스토리가 시간순에서도 유지·강화됨.

## 6. 재현성
- 모든 실험은 from-scratch 독립 학습(이전 런/체크포인트 비의존). recency는 RecBole train split에서만 추출(누수 없음, 코드로 검증).
- timestamp 복원·temporal split·recency 집계 모두 **LLM 무관 결정적**(seed 고정). intent facet만 저장된 LLM 출력(`step2_related_intents.csv`)에 의존.
- 재현:
  ```
  python add_timestamps.py && python make_temporal_inter.py
  IKGR_SPLIT=TO IKGR_EPOCHS=12 IKGR_SEEDS=2020,2021,2022 \
  IKGR_SPECS=IKGR_kgoff,IKGR_full_hetero,IKGR_dyn,IKGR_dyn_attn,BPR,LightGCN \
  python eval_slices.py    # -> run/slice_eval_TO_result.json
  ```

## 7. 한계 & 다음 단계
- **gain은 modest**(recency overall +3%): dense k=100 + per-user 시간 split이라 효과 폭이 제한적. 글로벌 시간 split(진짜 cold-start 유저)·sparse 코어(k=20/30)에서 더 클 가능성(LLM 비용 → Qwen 전환).
- recency τ/N 튜닝, crowds facet은 미탐색(보류).
- **다음: CORONA(3단계)** — 동적 KG 탐색→후보 생성→LLM 필터링→가중합 랭킹. IKGR+DynLLM 임베딩을 후보/스코어 소스로 재사용.

## 8. 산출물 / 커밋
- 코드: `goodreads_preprocess.py`(timestamp), `add_timestamps.py`, `make_temporal_inter.py`, `ikgr_core/model_ikgr.py`(use_dynamic/profile_attn), `eval_slices.py`(IKGR_SPLIT=TO + recency 빌더 + specs).
- 결과: `run/slice_eval_TO_result.json`. 설계: `DYNLLM_INTEGRATION.md`. 참조: `dynmLLM/`(공식).
- 커밋: `5817dc1`(Step1 TO), `388dbb2`(Step2 recency), `cffb94d`(Step3 attn 기각).
