from fastapi import FastAPI
from pydantic import BaseModel
from src.config import DATA_PATH
from src.ingestion import load_and_split
from src.vectorstore import build_vectorstore
from src.chain import build_chain

app = FastAPI()

# 서버 시작할 때 RAG 파이프라인 초기화
chunks = load_and_split(DATA_PATH)
vectorstore = build_vectorstore(chunks)
chain = build_chain(vectorstore)

# 요청/응답 모델
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def root():
    return {"message": "RAG API 서버 실행 중!"}

@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    result = chain.invoke({"query": request.question})
    return QueryResponse(question=request.question, answer=result)