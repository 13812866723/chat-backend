"""LLM 提供商工厂"""
from config.llm import llm_config
from providers.siliconflow import SiliconFlowProvider
from providers.zhipu import ZhipuProvider


def get_llm_provider():
    """获取大模型服务实例"""
    provider = llm_config.PROVIDER.lower()
    if provider == "siliconflow":
        return SiliconFlowProvider()
    elif provider == "zhipu":
        return ZhipuProvider()
    raise ValueError(f"不支持的 LLM provider: {llm_config.PROVIDER}")
