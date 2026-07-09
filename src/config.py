import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY가 설정되지 않았습니다. "
        ".env 파일을 확인하세요."
    )

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
DATA_PATH = "data/alex-notes"  # 여기 변경
VECTORSTORE_PATH = "data/faiss_index"

# Gemini free tier: embed_content 100 requests/min
EMBED_BATCH_SIZE = 80
EMBED_BATCH_DELAY_SEC = 65

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100