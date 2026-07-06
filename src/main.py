from src.graph import rag_graph

def main():
    result = rag_graph.invoke({"query": "RAG가 뭐야?"})
    print(f"질문: RAG가 뭐야?")
    print(f"답변: {result['answer']}")

if __name__ == "__main__":
    main()