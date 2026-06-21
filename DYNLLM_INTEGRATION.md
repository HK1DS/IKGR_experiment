# DynLLM 통합 설계 (Option B: 메커니즘 재구현)

졸업작품 2단계 **DynLLM**을, 공식 레포(`dynmLLM/`, TGN 기반 시간 그래프 추천)를 **그대로 이식하지 않고**,
그 핵심 아이디어를 우리 **RecBole 기반 IKGR 프레임워크 위에 component로 재구현**한다.
목적은 동일 데이터·split·평가에서 **`IKGR` vs `IKGR+DynLLM` vs `Full` ablation을 일관되게** 찍는 것.

---

## 0. 왜 이식(A)이 아니라 재구현(B)인가
- 공식 DynLLM은 TGN memory + 시간순 split + 자체 metric harness → **IKGR(RecBole 정적)과 같은 표에서 비교 불가**.
- 코드 미완성(`ProfilesHistoryUpdater` 미정의), data/embedding 미동봉, 4-facet 1536d 임베딩 신규 생성 필요.
- 우리는 이미 facet 자산을 보유 → 재구현이 싸고 일관적.

## 1. DynLLM → 우리 프레임워크 매핑

| DynLLM 구성요소 | 다이어그램 | 우리 재구현 | 신규 비용 |
|---|---|---|---|
| 4-facet 프로필 (crowds/interests/categories/brands) | 2-1 multi-faceted profile | **interests=IKGR intent**, **brands=author/publisher**, **categories=shelf**, (+attribute) | 0 (기보유) |
| | | **crowds** = 상호작용 기반 유저 군집(co-occurrence/clustering) | 0 (무료 파생) |
| multi_profile_target (MHA로 프로필→유저 융합) | 2-1 | **facet attention fusion** 모듈 | 코드만 |
| memory + temporal attention + time projection (동적 갱신) | 2-2 edge reweighting | **recency-weighted facet aggregation** (시간감쇠 가중) | timestamp 필요 (→ (b)) |
| BPR loss / full-sort eval | - | 그대로 (RecBole) | 0 |

**핵심 단순화:** DynLLM의 "동적 KG 엣지 가중치 갱신"(2-2)을, 우리 정적 프레임워크에서는
**"시간 감쇠로 유저의 facet 집계 가중치를 동적으로 조절"**로 해석한다 (최근 상호작용/의도가 더 큰 가중).
TGN memory 전체를 옮기지 않고도 "동적성"의 핵심(최신성 반영)을 얻는다.

## 2. 데이터 요구사항
- **timestamp 필요.** 현재 `interactions_k100.csv`엔 `user_id,item_id,rating`만 있음.
- 원본 `goodreads_interactions_children.json.gz`에 `date_added/read_at/started_at` 존재(확인됨) → **(b)에서 복원**.
- 산출 목표: `interactions_k100.csv`에 `timestamp` 컬럼 추가 (epoch 초 or 정렬 가능한 값).
- crowds facet: 유저별 상호작용 아이템 집합으로 군집(예: 아이템 shelf 분포 기반 KMeans, 또는 co-occurrence)→ 유저→crowd 노드 엣지. (무료, 메타데이터/상호작용만 사용)

## 3. 모델 component 설계 (`ikgr_core/model_ikgr.py` 확장)

기존 `_emb_users`/`_emb_items`의 facet gather 위에 두 가지를 추가:

### 3.1 Recency-weighted facet aggregation (동적성)
현재: 유저의 intent/facet을 **균등 평균**(`_agg`)으로 집계.
변경: 유저-노드 엣지에 **시간 감쇠 가중** `w = exp(-Δt / τ)` (Δt = eval 시점 - 상호작용 시각, τ 학습/고정).
→ 최근에 형성된 의도/카테고리가 유저 표현을 더 지배 = "동적 프로필".
- 구현: padded gather에 per-edge timestamp를 같이 저장, 집계 시 softmax(recency) 가중 평균.
- `use_dynamic` 토글로 on/off (ablation: 정적 평균 vs recency 가중).

### 3.2 Multi-facet attention fusion (DynLLM multi_profile_target)
현재: facet 기여를 **학습 스칼라 게이트 α**로 단순 합산.
변경: 유저 base 임베딩을 query, 각 facet 집계 벡터(intent/brand/category/attr/crowd)를 key/value로
**MultiheadAttention** → facet별 가중을 유저·맥락에 따라 동적 배분.
- `use_profile_attention` 토글.
- 출력 = base ⊕ attn(facets) → 최종 유저 임베딩.

### 3.3 토글 매트릭스 (ablation 설계)
| 구성 | use_kg | use_meta_kg | use_dynamic | profile_attn |
|---|---|---|---|---|
| MF | ✗ | ✗ | ✗ | ✗ |
| IKGR (현재) | ✓ | ✓ | ✗ | ✗ |
| **IKGR+DynLLM** | ✓ | ✓ | ✓ | ✓ |
| (분해) +recency만 | ✓ | ✓ | ✓ | ✗ |
| (분해) +attn만 | ✓ | ✓ | ✗ | ✓ |

## 4. 평가 프로토콜
- **시간순 split으로 전환**(RecBole `eval_args order=TO`, leave-last-by-time). DynLLM 스테이지는 시간성이 핵심이므로 TO가 맞음.
- **공정성:** IKGR baseline도 동일 TO split에서 재실행(기존 RS 표와 별도 섹션).
- 지표: 기존대로 NDCG/Recall/MRR/Hit/Precision@{10,30} + coverage/novelty + cold-start/long-tail 슬라이스.
- `eval_slices.py`에 spec 추가: `IKGR_dyn`, `IKGR_dyn_attn`, `IKGR_full_dyn` 등.

## 5. 구현 단계
1. **(b) timestamp 복원** — `goodreads_preprocess.py`/`apply_k_core.py` 시간 인지화, 또는 기존 k-core에 원본 join으로 backfill.
2. **crowds facet 빌더** — `build_crowds.py` (유저 군집 → user→crowd 엣지 pack). 무료.
3. **모델 확장** — recency-weighted `_agg` + profile-attention fusion + 토글.
4. **eval 확장** — TO split + 새 spec, 다중시드.
5. **ablation 표 + 리포트** — `IKGR_REPORT.md`/`DynLLM` 섹션.

## 6. 비용 / 리스크
- **추가 LLM 비용 0** (intent·meta 임베딩 재사용, crowds·timestamp 무료).
- 리스크: TO split에서 IKGR/baseline 수치가 RS와 달라짐(정상, 별도 비교). recency τ 튜닝 필요. attention fusion 과적합 가능 → 다중시드로 검증.
- 공식 DynLLM의 "정확한 재현"은 본 설계 범위 밖(참조용). 본 작업은 "DynLLM 아이디어의 프레임워크 내 통합".

## 7. 산출물(예정)
- `build_crowds.py`, 확장된 `ikgr_core/model_ikgr.py`, `eval_slices.py` spec, timestamp 포함 데이터, 결과 json, 리포트 섹션.
