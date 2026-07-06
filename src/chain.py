from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from src.config import LLM_MODEL

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""
아래 context를 바탕으로 질문에 답해주세요.

Context: {context}

질문: {question}
""")

    def retrieve(query: str):
        return vectorstore.as_retriever().invoke(query)

    chain = (
        {
            "context": itemgetter("query") | RunnableLambda(retrieve) | format_docs,
            "question": itemgetter("query")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG 체인 구성 완료!")
    return chain