from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
    아래 context를 바탕으로 질문에 답해주세요.

    Context: {context}

    질문: {question}
""")