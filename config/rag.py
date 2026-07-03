"""RAG 配置"""
import os
from dotenv import load_dotenv

load_dotenv()


class RAGConfig:
    """RAG 服务配置"""
    EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))
    TOP_K = int(os.getenv("RAG_TOP_K", "3"))
    CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    DOCUMENTS_DIR = os.getenv("RAG_DOCUMENTS_DIR", "./documents")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")


class EmbeddingConfig:
    """嵌入模型配置"""
    API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", ""))
    BASE_URL = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"))
    MODEL = os.getenv("RAG_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
    TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "60"))


rag_config = RAGConfig()
embedding_config = EmbeddingConfig()
