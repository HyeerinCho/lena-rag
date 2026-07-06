from src.graph import rag_graph

def main():
    result = rag_graph.invoke({"query": "RAG가 뭐야?"})
    print(f"질문: RAG가 뭐야?")
    print(f"답변: {result['answer']}")

if __name__ == "__main__":
    main()



"""
from src.config import DATA_PATH
from src.ingestion import load_and_split
from src.vectorstore import load_or_build_vectorstore
from src.chain import build_chain

def main():
    chunks = load_and_split(DATA_PATH)
    vectorstore = load_or_build_vectorstore(chunks)
    chain = build_chain(vectorstore)

    question = "RAG가 뭐야?"
    result = chain.invoke({"query": question})
    print(f"질문: {question}")
    print(f"답변: {result}")

if __name__ == "__main__":
    main()
"""