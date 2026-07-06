import os
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 1. 문서 로딩
def load_document(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

# 2. 청킹
def split_text(text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# 3. 임베딩
def embed_texts(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    return result["embedding"]

# 4. 유사도 검색
def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query: str, chunks: list[str], chunk_embeddings: list[list[float]], top_k: int = 2) -> list[str]:
    query_embedding = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    scores = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# 5. 생성
def generate(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
아래 context를 바탕으로 질문에 답해주세요.

Context: {context}

질문: {query}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# 메인 실행
if __name__ == "__main__":
    # 1. 로딩
    text = load_document("data/sample.txt")
    print(f"문서 로딩 완료!")

    # 2. 청킹
    chunks = split_text(text)
    print(f"청킹 완료: {len(chunks)}개 조각")

    # 3. 임베딩
    chunk_embeddings = embed_texts(chunks)
    print(f"임베딩 완료!")

    # 4. 검색 + 생성
    question = "RAG가 뭐야?"
    context_chunks = retrieve(question, chunks, chunk_embeddings)
    answer = generate(question, context_chunks)

    print(f"질문: {question}")
    print(f"답변: {answer}")