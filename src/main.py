import sys

from src.rag import ask


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "RAG가 뭐야?"
    print(f"질문: {query}")
    print(f"답변: {ask(query)}")


if __name__ == "__main__":
    main()