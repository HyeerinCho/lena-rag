from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import EMBEDDING_MODEL
import os

FAISS_PATH = "data/faiss_index"

def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 저장된 벡터DB 있으면 로딩, 없으면 새로 만들기
    if os.path.exists(FAISS_PATH):
        print("저장된 벡터DB 로딩...")
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("벡터DB 새로 생성 중...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_PATH)
        print("벡터DB 저장 완료!")

    return vectorstore