# LENA-PJ — 과제 회고

개인 학습 노트(`alex-notes`)를 지식 베이스로 활용하는 **LangGraph 기반 RAG 파이프라인** 프로젝트입니다.

---

## 1. 과제 목표

- LangChain + **LangGraph** 기반 RAG 파이프라인 구축
- FastAPI로 REST API 래핑
- LangSmith로 체인 실행 Tracing 및 Dataset 기반 평가 (진행 중)

---

## 2. 버전별 변경 사항 요약

| 버전 | 핵심 변경 | 결과 | 날짜 |
|------|-----------|------|------|
| v0 | `plain_rag.py` — LangChain 없이 Google SDK로 RAG 직접 구현 (로딩 → 청킹 → 임베딩 → 코사인 유사도 검색 → 생성) | RAG 원리 이해, 단일 파일 프로토타입 | 7/初 |
| v1 | alex-rag 데모 참고, LangChain 스택 도입 (`ingestion` → `vectorstore` → `chain` / `graph` 분리), `data/alex-notes` 코퍼스 연결 | 기반 구조 완성, but CLI(graph)와 API(chain) 이원화 | 7/初 |
| v2 | **Phase 1 리팩터링** — `prompts.py` 프롬프트 분리, `rag.py` + `ask()` 단일 진입점, LangGraph를 RAG 코어로 통일, `chain.py` 제거 | CLI·API가 동일한 LangGraph 파이프라인 공유 | 7/9 |
| v3 | **Phase 2 부분 완료** — `config.py` API 키 검증, `VECTORSTORE_PATH` 일원화, `CHUNK_SIZE`/`CHUNK_OVERLAP` config 이동 | 설정 중복 제거, 시작 시 에러 명확화 | 7/9 |

---

## 3. 핵심 개념 정리

### 3-1. RAG 흐름

```
사용자 입력
    ↓
[ingestion] data/alex-notes/*.md 로딩 → RecursiveCharacterTextSplitter 청킹 (500자, overlap 100)
    ↓
[vectorstore] Gemini embedding → FAISS 인덱스 (data/faiss_index/ 디스크 캐싱)
    ↓
[LangGraph — retrieve 노드] 질문과 유사한 문서 청크 검색
    ↓
[LangGraph — generate 노드] context + prompt → Gemini 답변 생성
    ↓
답변 반환
```

### 3-2. 모델 구성 (현재)

| 역할 | 모델 |
|------|------|
| 임베딩 | Google `gemini-embedding-001` |
| 답변 생성 | Google `gemini-2.5-flash` |
| 벡터 DB | FAISS (로컬 파일) |
| 평가 (예정) | LangSmith Dataset `lena-rag` |

### 3-3. 파일 구조 (현재 v3)

| 파일 | 역할 |
|------|------|
| `data/alex-notes/*.md` | RAG 검색 대상 — 개인 학습·일기·프로젝트 노트 |
| `src/config.py` | 환경변수, 모델명, 경로, 청킹 설정 |
| `src/ingestion.py` | 마크다운 로딩 + 텍스트 청킹 |
| `src/vectorstore.py` | FAISS 생성/로드 (디스크 캐싱) |
| `src/prompts.py` | RAG 프롬프트 템플릿 |
| `src/graph.py` | LangGraph RAG — `retrieve` → `generate` 노드 |
| `src/rag.py` | lazy init + `ask(query)` 단일 API |
| `src/main.py` | CLI — `lena` 명령 진입점 |
| `src/api.py` | FastAPI — `POST /query` 엔드포인트 |
| `eval/dataset.py` | LangSmith Dataset 생성 및 예제 시드 |
| `scripts/lena` | CLI 래퍼 |
| `scripts/lena-api` | Uvicorn 서버 래퍼 |

### 3-4. 실행 흐름 (CLI 기준)

```
사용자가 lena "LangChain이 뭐야?" 실행
    ↓
scripts/lena → src/main.py
    ↓
src/rag.py — ask(query)
    ↓
src/rag.py — get_rag_graph()  [최초 1회만 실행]
    ├── ingestion.load_and_split(DATA_PATH)  → 45개 문서, 387개 청크
    ├── vectorstore.build_vectorstore(chunks)  → FAISS 로드 또는 생성
    └── graph.build_rag_graph(vectorstore)  → LangGraph 컴파일
    ↓
graph — retrieve 노드
    ├── retriever.invoke(query)  → 유사 문서 k개 검색
    └── state["documents"] 업데이트
    ↓
graph — generate 노드
    ├── context = 문서 청크 join
    ├── RAG_PROMPT + gemini-2.5-flash → 답변 생성
    └── state["answer"] 업데이트
    ↓
ask() — result["answer"] 반환 → 터미널 출력
```

### 3-5. 실행 흐름 (FastAPI 기준)

```
사용자가 POST /query {"question": "RAG가 뭐야?"} 전송
    ↓
src/api.py — query()
    ↓
src/rag.py — ask(question)  [CLI와 동일한 파이프라인]
    ↓
QueryResponse(question, answer) JSON 응답
```

---

## 4. 겪었던 문제와 해결

| 문제 | 원인 | 해결 | 파일 |
|------|------|------|------|
| `python` 명령 not found | `.python-version`이 3.14인데 pyenv에는 3.11만 설치 | `source .venv/bin/activate` 또는 `uv run` 사용 | `.python-version`, `.venv` |
| `build_rag_graph() takes 0 positional arguments` | `graph.py`에 새/옛 `build_rag_graph` 함수가 중복 정의, 나중 정의가 덮어씀 | 41행 이후 예전 코드 삭제 | `src/graph.py` |
| CLI와 API가 다른 RAG 사용 | `main.py`는 graph, `api.py`는 chain 사용 | `ask()` 단일 진입점으로 통일, `chain.py` 제거 | `src/rag.py`, `src/api.py` |
| import 시마다 문서 로딩·인덱싱 | `graph.py` 모듈 레벨에서 `load_and_split` 실행 | `@lru_cache` + `get_rag_graph()`로 lazy init | `src/rag.py` |
| 매번 인덱싱 (초기) | 벡터를 RAM에만 저장 | FAISS `save_local` / `load_local`로 디스크 캐싱 | `src/vectorstore.py` |
| API 키 없을 때 불명확한 에러 | `GOOGLE_API_KEY` None 검증 없음 | 시작 시 `ValueError` 발생 | `src/config.py` |
| 설정 상수 중복 | `VECTORSTORE_PATH`와 `FAISS_PATH` 별도 정의 | `config.VECTORSTORE_PATH`로 일원화 | `src/config.py`, `src/vectorstore.py` |
| Gemini API 429 (예상) | free tier embed 100 req/min | `EMBED_BATCH_SIZE`/`EMBED_BATCH_DELAY_SEC` config에 준비 (미적용) | `src/config.py` |

---

## 5. 실행 방법

### 환경 설정

```bash
cp .env.example .env
# .env에 GOOGLE_API_KEY 입력 (필수)
```

### 의존성 설치

```bash
uv sync
source .venv/bin/activate   # 또는 매번 uv run 사용
```

### CLI

```bash
lena                        # 기본 질문: "RAG가 뭐야?"
lena "LangChain이 뭐야?"    # 질문 지정
```

### API 서버

```bash
lena-api
# 다른 터미널
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG가 뭐야?"}'
```

### 벡터 인덱스 재생성 (노트 수정 후)

```bash
rm -rf data/faiss_index/
lena "테스트 질문"
```

### LangSmith Dataset 시드

```bash
uv run python eval/dataset.py
```

---

## 6. 회고

### 6-1. 과정 서술

7주차 과제는 LangChain 기반 RAG 파이프라인 구축, FastAPI 래핑, LangSmith 평가였다. 수업에서 alex가 보여준 [alex-rag](https://github.com/100-hours-a-week/alex-rag) 데모를 참고해 개인 학습 노트(`alex-notes`)를 코퍼스로 삼는 RAG를 만들기로 했다.

**[v0]** 먼저 `plain_rag.py`로 LangChain 없이 Google SDK만 써서 로딩 → 청킹 → 임베딩 → 코사인 유사도 → 생성 흐름을 직접 구현해 봤다. RAG가 어떤 단계로 이루어지는지 머릿속에 그려지는 데 도움이 됐지만, 프레임워크 없이는 보일러플레이트가 많았다.

**[v1]** LangChain 스택을 도입했다. `ingestion.py`, `vectorstore.py`, `graph.py`(LangGraph), `chain.py`(LCEL)로 파일을 나눴고, FAISS 디스크 캐싱으로 매번 인덱싱하는 문제를 해결했다. CLI는 LangGraph, API는 LCEL chain을 쓰는 이원화 상태가 됐고, 프롬프트도 두 곳에 중복됐다.

**[v2]** Phase 1 리팩터링을 진행했다. `prompts.py`로 프롬프트를 분리하고, `rag.py`의 `ask()`를 CLI·API 공통 진입점으로 만들었다. LangGraph를 RAG 코어로 확정하고 `chain.py`를 제거했다. `graph.py`에 예전 코드가 남아 `TypeError`가 났던 것도 직접 디버깅하며 해결했다.

**[v3]** Phase 2로 설정을 정리했다. API 키 검증, 경로 상수 일원화, 청킹 파라미터를 `config.py`로 모았다. `lena "LangChain이 뭐야?"`로 end-to-end 동작을 확인했다.

### 6-2. 느낀점

Week 7 감정 분류 과제(v1~v5)에서 Ollama 임베딩 모델(`nomic-embed-text` vs `bge-m3`)에 따라 결과가 크게 달라진 경험이 있었다. 이번 LENA-PJ에서도 **어떤 데이터를 넣느냐, 파이프라인을 어떻게 구성하느냐**에 따라 답변 품질이 달라진다는 걸 다시 느꼈다.

직접 타이핑하고 리팩터링하면서, AI에게 "100을 입력해야 100이 나온다"는 말이 코드 구조에도 그대로 적용된다는 걸 알게 됐다. 프롬프트만이 아니라 **파일 분리, 초기화 시점, 단일 진입점** 같은 설계 선택도 출력 품질과 유지보수에 영향을 준다.

LangGraph로 retrieve → generate를 노드로 나누니 각 단계 로그를 보며 디버깅하기 쉬웠고, 나중에 relevance grading이나 재검색 노드를 추가하기도 좋겠다는 생각이 들었다.

### 6-3. 앞으로 할 일

- [ ] `pyproject.toml` 미사용 의존성 정리 (`google-generativeai`, `numpy`, `pypdf`, `ragas`)
- [ ] `eval/dataset.py` — `if __name__ == "__main__"` 가드, 중복 예제 방지
- [ ] LangSmith Dataset 기반 자동 평가 파이프라인
- [ ] retriever `top_k` 튜닝, 빈 검색 fallback
- [ ] FAISS 인덱스 무효화 전략 (`--reindex` 옵션 등)
- [ ] LangGraph 확장 — 출처 표시, relevance grading 노드

---

## 7. 참고 링크

- [alex-rag (수업 데모)](https://github.com/100-hours-a-week/alex-rag)
- [LangChain 문서](https://python.langchain.com/)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/)
