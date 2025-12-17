# OurHistoryNet__HyperCLOVA_X_SEED_1.5B
한국사 QA 스타일의 **SFT(LoRA) + 간단 RAG(TF-IDF)** 실습용 파이프라인.

1. 우리역사넷(신편 한국사) 페이지를 크롤링하여 원문을 JSONL로 저장
2. 원문을 SFT 학습용 JSONL로 변환(긴 글 chunking)
3. HyperCLOVAX-SEED-Text-Instruct-1.5B 기반 LoRA(fp16) 파인튜닝
4. LoRA 어댑터 + TF-IDF 검색 기반 RAG로 CLI 채팅

---

## Repository layout

* `01_crawl_nh_raw.py`: 크롤러 (출력: `nh_raw.jsonl`) 
* `02_make_history_sft.py`: SFT 데이터 생성기 (출력: `history_sft_train.jsonl`) 
* `03_train_clovax_lora_fp16.py`: LoRA(fp16) 학습 (출력 폴더: `./HyperCLOVAX-history-lora-fp16`) 
* `chat_clovax_history_lora_rag.py`: TF-IDF RAG + LoRA 어댑터로 채팅 

---

## Requirements

* Python 3.x
* (권장) CUDA GPU (LoRA fp16 학습/추론)
* 주요 라이브러리:

  * `requests`, `beautifulsoup4`, `tqdm` (크롤링/진행바) 
  * `datasets`, `transformers`, `peft`, `torch` (학습/추론) 
  * `scikit-learn` (TF-IDF + cosine similarity) 

설치:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install requests beautifulsoup4 tqdm datasets transformers peft accelerate scikit-learn
# torch는 환경(CUDA/OS)에 맞는 방식으로 설치하세요.
```

---

## Quickstart

### 1) Crawl raw documents

```bash
python 01_crawl_nh_raw.py
```

* 기본 시작 URL: `START_URL = "https://contents.history.go.kr/front/nh/view.do?levelId=nh_009_0020"` 
* 출력: `nh_raw.jsonl` 
* 과도한 크롤링 방지를 위해 `MAX_PAGES = 20` 제한이 걸려 있음. 
* 서버 부하 완화를 위해 페이지당 `time.sleep(0.4)` 지연이 들어가 있음. 

---

### 2) Build SFT dataset (JSONL)

```bash
python 02_make_history_sft.py
```

* 입력: `nh_raw.jsonl` 
* 출력: `history_sft_train.jsonl` 
* 긴 글은 `MAX_CHARS` 기준으로 줄/문단 단위로 chunking 함. 
* 현재 SFT 레코드는 **`output = input`(restoration/mimetic SFT)** 형태. 

---

### 3) Train LoRA (fp16)

```bash
python 03_train_clovax_lora_fp16.py
```

기본 설정:

* Base model: `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` 
* Train data: `history_sft_train.jsonl` 
* Output dir: `./HyperCLOVAX-history-lora-fp16` 
* `MAX_SEQ_LEN = 4096` 
* LoRA target modules: `q_proj, k_proj, v_proj, ...` 
* 메모리 절약: gradient checkpointing 활성화 + `use_cache=False` 
* 학습 하이퍼파라미터(예): bs=1, grad_acc=8, epochs=2, lr=2e-4, fp16=True 

---

### 4) Chat with LoRA + TF-IDF RAG

```bash
python chat_clovax_history_lora_rag.py
```

* 문서 로드: `nh_raw.jsonl` 
* 생성 파라미터(예): `max_new_tokens=4096, top_p=0.9, temperature=0.8` 
* 종료: `exit` 또는 `quit` 입력 

---

## Acknowledgements

* Base model: `HyperCLOVAX-SEED-Text-Instruct-1.5B` (NAVER) 
* Data source: `contents.history.go.kr` (우리역사넷/신편 한국사) 
