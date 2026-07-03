"""LLM 提供商抽象基类"""
from abc import ABC, abstractmethod
from typing import List


class BaseLLMProvider(ABC):
    """大模型服务基类"""


    @abstractmethod    
    # 强制规范
    def chat(self, messages: List[dict], **kwargs) -> str:
        """发送对话请求，返回 assistant 的回复"""
        pass

    @abstractmethod
    def chat_stream(self, messages: List[dict], **kwargs):
        """流式对话请求"""
        pass
