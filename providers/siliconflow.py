"""SiliconFlow 大模型服务"""
import requests
from langchain_openai import ChatOpenAI
from providers.base import BaseLLMProvider
from config.llm import llm_config


class SiliconFlowProvider(BaseLLMProvider):
    """SiliconFlow 大模型服务"""

    def __init__(self):
        self.api_key = llm_config.API_KEY
        self.model = llm_config.MODEL
        self.base_url = llm_config.BASE_URL

    def bind_tools(self, tools: list, **kwargs):
        """
        绑定工具列表，返回一个已绑定工具的 LangChain ChatOpenAI 实例。

        SiliconFlow 兼容 OpenAI API，因此使用 ChatOpenAI 作为 LangChain 模型包装器。

        参数:
            tools: 工具列表（LangChain BaseTool 实例）
            **kwargs: 其他传递给 bind_tools 的参数

        返回:
            绑定了工具的 ChatOpenAI 实例
        """
        llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=0,
        )
        return llm.bind_tools(tools, **kwargs)

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
        response.raise_for_status()  # 检查响应状态码是否正常
        for line in response.iter_lines(): 
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
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
