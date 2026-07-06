from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from src.vectorstore import build_vectorstore
from src.ingestion import load_and_split
from src.config import DATA_PATH, LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. State 정의 (그래프에서 노드 간에 공유되는 데이터)
class RAGState(TypedDict):
    query: str
    documents: List[Document]
    answer: str

# 2. 벡터스토어 로딩
chunks = load_and_split(DATA_PATH)
vectorstore = build_vectorstore(chunks)
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

# 3. 노드 함수 정의

# 노드 1: 검색
def retrieve(state: RAGState) -> RAGState:
    print(f"[검색 노드] 쿼리: {state['query']}")
    documents = retriever.invoke(state["query"])
    return {"documents": documents}

# 노드 2: 생성
def generate(state: RAGState) -> RAGState:
    print(f"[생성 노드] 문서 {len(state['documents'])}개로 답변 생성 중...")
    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = ChatPromptTemplate.from_template("""
아래 context를 바탕으로 질문에 답해주세요.

Context: {context}

질문: {question}
""")
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["query"]})
    return {"answer": answer}

# 4. 그래프 구성
def build_rag_graph():
    graph = StateGraph(RAGState)

    # 노드 추가
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    # 엣지 연결
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag_graph = build_rag_graph()