# IKGR Pipeline Progress & Execution Guide

이 문서는 IKGR (Intent Knowledge Graph Recommender) 파이프라인의 진행 상황 및 향후 프로젝트를 새로 내려받아 이어서 진행할 때 참고할 수 있는 실행 가이드라인을 정리한 파일입니다.

---

## 1. 현재 진행 상황 및 이슈 요약

### 📊 데이터 준비 상황
* 원본 데이터(`goodreads_books_children.json.gz`, `goodreads_interactions_children.json.gz`)를 CSV로 파싱하는 전처리를 마쳤습니다.
* 원본 전체 데이터(38.6GB)는 로컬 테스트 및 API 비용 절감을 위해 **2,138행 크기의 샘플 데이터셋**(`profiles_sample.csv`, `interactions_sample.csv`)으로 추출해 두었습니다.

### 🛑 진행 중단 사유 (Gemini API 쿼터 초과)
* `step1.py` (유저/아이템 의도 추출)를 실행하던 중, **Gemini 3.0 Flash Free Tier API**의 호출 한도에 걸려 중단되었습니다.
* **에러 메시지**: `429 RESOURCE_EXHAUSTED (Quota exceeded for generate_content_free_tier_requests)`
* **원인**: 2,138행의 샘플이라도 첫 실행 시 유저/아이템 각각 API를 요청하므로 약 4,200회 이상의 호출이 필요한데, 무료 티어의 제한(분당 15회 및 일일 1,500회)으로 인해 대기 시간이 무한정 길어지다 에러가 발생해 멈췄습니다.

### 🧹 깃허브 업로드 준비 완료
* 대용량 원본 데이터(`data/`) 및 로컬 캐시/출력 폴더(`run/`)는 `.gitignore`에 등록하여 깃허브 업로드 대상에서 배제했습니다.
* 임시로 사용했던 테스트용 스크립트(`test_llm.py`, `explore_data.py` 등) 및 로컬 캐시 파일들은 모두 삭제하여 리포지토리를 깔끔하게 정리했습니다.

---

## 2. 프로젝트를 새로 받아 이어서 진행하는 방법

다른 환경이나 깃허브에서 프로젝트를 다시 클론(Clone)받아 진행할 때의 단계별 순서입니다.

### Step 1. 가상환경 설정 및 패키지 설치
리포지토리 폴더(`IKGR-main/IKGR-main`) 내에서 가상환경을 생성하고 필요한 라이브러리를 설치합니다.
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (Windows PowerShell 기준)
.venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### Step 2. API Key 및 환경 설정 (.env 생성)
1. 리포지토리 루트에 있는 `.env.example` 파일을 복사해 `.env` 파일을 생성합니다.
2. `.env` 파일 안에 본인의 API Key를 입력합니다.
   ```bash
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### Step 3. 데이터 전처리 (JSON.GZ ➡️ CSV 변환)
만약 `data/` 폴더에 CSV 파일들이 없고 원본 `.json.gz` 파일만 존재하는 상태라면, 전처리 스크립트를 실행하여 파이프라인용 CSV를 먼저 추출해야 합니다.
```bash
python goodreads_preprocess.py \
  --interactions_gz goodreads_interactions_children.json.gz \
  --books_gz goodreads_books_children.json.gz \
  --out_profiles data/profiles.csv \
  --out_interactions data/interactions.csv
```
* **출력 파일**: `data/profiles.csv` (약 38GB) 및 `data/interactions.csv` (약 450MB)가 생성됩니다.

### Step 4. 테스트용 샘플 데이터 추출 (선택 사항)
전체 데이터를 LLM API로 처리하기엔 시간과 비용이 막대하므로, 테스트용 소형 샘플을 만드는 것이 좋습니다. (현재 환경에는 2,138행으로 구성된 샘플이 이미 생성되어 `data/`에 저장되어 있습니다.)

만약 새로 샘플링을 진행하려면 아래와 같이 파이썬 코드를 통해 앞부분의 일부 행만 슬라이싱하여 샘플 파일을 생성할 수 있습니다.
```python
import pandas as pd

# interactions 샘플 생성 (예: 상위 2000개만 사용)
df_inter = pd.read_csv("data/interactions.csv", nrows=2000)
df_inter.to_csv("data/interactions_sample.csv", index=False)

# interactions 샘플에 포함된 user_id, item_id에 매칭되는 profile만 추출
u_ids = set(df_inter["user_id"])
i_ids = set(df_inter["item_id"])

df_prof = pd.read_csv("data/profiles.csv")
df_prof_sample = df_prof[df_prof["user_id"].isin(u_ids) & df_prof["item_id"].isin(i_ids)]
df_prof_sample.to_csv("data/profiles_sample.csv", index=False)
```

### Step 5. config.yaml 설정 확인
* `config.yaml` 파일의 `paths` 섹션이 실행하려는 데이터 경로를 가리키고 있는지 확인합니다:
  * **전체 데이터로 실행 시**:
    ```yaml
    paths:
      input_csv: data/profiles.csv
      inter_file: data/interactions.csv
    ```
  * **샘플 데이터로 테스트 시**:
    ```yaml
    paths:
      input_csv: data/profiles_sample.csv
      inter_file: data/interactions_sample.csv
    ```

### Step 6. LLM API 변경 및 재시도 (중요 💡)
가성비가 좋은 Qwen(Alibaba)이나 DeepSeek, 혹은 제공받은 Luxia Cloud API나 유료 OpenAI/Gemini API로 전환하여 `step1`을 이어서 실행합니다.

* **Luxia Cloud API (OpenAI GPT-4o-mini Bridge)로 변경하는 경우**:
  `config.yaml`의 `llm` 설정을 다음과 같이 변경합니다. (.env 파일에 `LUXIA_API_KEY` 입력 필수)
  ```yaml
  llm:
    base_url: https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini
    api_key: "${LUXIA_API_KEY}"
    model: "llm"
    provider: luxia
    temperature: 0.2
    top_p: 0.95
    max_tokens: 2048
  ```

* **OpenAI 호환 API (Qwen, DeepSeek 등)로 변경하는 경우**:
  `config.yaml`의 `llm` 설정을 다음과 같이 변경합니다.
  ```yaml
  llm:
    base_url: "https://api.deepseek.com/v1" # 또는 사용하려는 모델 API의 Endpoint
    api_key: "${DEEPSEEK_API_KEY}"          # .env에 키 정의 필요
    model: "deepseek-chat"
    provider: openai                        # openai로 변경하면 표준 OpenAI API 규격으로 통신합니다.
    temperature: 0.2
    max_tokens: 2048
  ```

### Step 7. 파이프라인 단계별 실행

1. **Step 1: 유저 및 아이템 고유 의도 추출**
   ```bash
   python step1.py
   ```
   * 완료 시 `run/step1_intents.csv` 파일이 생성됩니다.

2. **Step 2: 추출된 의도를 RAG 및 LLM을 사용해 관련 의도로 확장**
   ```bash
   python step2.py
   ```
   * 완료 시 `run/step2_related_intents.csv` 파일이 생성됩니다.

3. **RecBole 연동용 데이터 포맷 변환 및 의도 임베딩 은행(Intent Bank) 구축**
   ```bash
   python convert_to_recbole_atomic.py --interactions data/interactions_sample.csv --intents run/step2_related_intents.csv --out_dir data --dataset ikgr-custom
   python build_intent_banks.py --step2_csv run/step2_related_intents.csv --encoder sentence-transformers/all-mpnet-base-v2 --user_out run/user_bank.pt --item_out run/item_bank.pt
   ```

4. **Step 3: IKGR 추천 GNN 모델 학습 및 평가**
   ```bash
   python step3.py
   ```
