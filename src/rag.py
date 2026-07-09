from functools import lru_cache

from src.config import DATA_PATH
from src.graph import build_rag_graph
from src.ingestion import load_and_split
from src.vectorstore import build_vectorstore


@lru_cache
def get_rag_graph():
    chunks = load_and_split(DATA_PATH)
    vectorstore = build_vectorstore(chunks)
    return build_rag_graph(vectorstore)


def ask(query: str) -> str:
    result = get_rag_graph().invoke({"query": query})
    return result["answer"]