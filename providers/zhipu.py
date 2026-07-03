"""智谱 AI 大模型服务"""
import json
import requests
from providers.base import BaseLLMProvider
from config.llm import llm_config


class ZhipuProvider(BaseLLMProvider):
    """智谱 AI 大模型服务"""

    def __init__(self):
        self.api_key = llm_config.API_KEY
        self.model = llm_config.MODEL
        self.base_url = llm_config.BASE_URL

    def chat(self, messages: list[dict], **kwargs) -> str:
        """发送对话请求"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        response = requests.post(url, headers=headers, json=payload, timeout=llm_config.TIMEOUT)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], **kwargs):
        """流式对话请求"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=llm_config.TIMEOUT)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        # choices 可能为空列表（usage 帧 / 结束帧），需防御空索引
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
