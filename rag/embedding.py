import requests
from langchain.embeddings.base import Embeddings
from typing import List
from config.rag import embedding_config


class SiliconFlowEmbeddings(Embeddings):
    """基于硅基流动 API 的嵌入模型"""

    def __init__(self, model: str = None):
        self.api_key = embedding_config.API_KEY
        self.model =  embedding_config.MODEL
        self.base_url = embedding_config.BASE_URL.rstrip("/")
        self.timeout = embedding_config.TIMEOUT  

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对多个文本进行嵌入"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        # 按输入顺序返回嵌入向量
        embeddings = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """对单个查询文本进行嵌入"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": text
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
