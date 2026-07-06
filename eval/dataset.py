import os

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

# LangSmith client uses LANGSMITH_API_KEY / LANGSMITH_ENDPOINT from .env
client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"),
)

DATASET_NAME = "lena-rag"

examples = [
    {
        "input": {"query": "RAG가 뭐야?"},
        "output": {
            "answer": "RAG는 Retrieval Augmented Generation의 약자로, 외부 문서에서 관련 정보를 검색하여 LLM의 답변 품질을 높이는 기법입니다."
        },
    },
    {
        "input": {"query": "LangChain이 뭐야?"},
        "output": {
            "answer": "LangChain은 LLM 기반 애플리케이션을 만들기 위한 프레임워크입니다."
        },
    },
    {
        "input": {"query": "LangSmith는 어떤 도구야?"},
        "output": {
            "answer": "LangSmith는 LangChain 체인의 실행을 모니터링하고 평가하는 도구입니다."
        },
    },
]


def ensure_dataset(name: str):
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        return client.create_dataset(dataset_name=name)


ensure_dataset(DATASET_NAME)

for example in examples:
    client.create_example(
        dataset_name=DATASET_NAME,
        inputs=example["input"],
        outputs=example["output"],
    )

print(f"Dataset '{DATASET_NAME}'에 {len(examples)}개 예제 추가 완료!")
