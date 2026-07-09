from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import ask

app = FastAPI()

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
    answer = ask(request.question)
    return QueryResponse(question=request.question, answer=answer)
