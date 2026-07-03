"""LLM 配置"""
import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """大模型配置"""
    PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    API_KEY = os.getenv("LLM_API_KEY", "")
    BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))


llm_config = LLMConfig()
