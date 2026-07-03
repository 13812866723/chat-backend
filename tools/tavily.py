"""Tavily 在线搜索工具模块"""
import os
from typing import List
from langchain_core.tools import BaseTool, tool

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@tool
def tavily_search_tool(query: str, max_results: int = 5) -> str:
    """
    Tavily 在线搜索工具 - 用于实时网络搜索。

    当需要获取最新信息、实时新闻、天气、股价、汇率、通用网络知识等
    不在本地知识库中的实时信息时使用此工具。

    参数:
        query: 搜索查询关键词
        max_results: 最大返回结果数量，默认 5 条

    返回:
        搜索结果摘要，包含标题、URL 和内容概要
    """
    if not TAVILY_API_KEY:
        return "错误：未配置 TAVILY_API_KEY 环境变量，无法使用在线搜索功能。"

    try:
        from langchain_tavily import TavilySearch

        search = TavilySearch(max_results=max_results, tavily_api_key=TAVILY_API_KEY)
        results = search.invoke({"query": query})

        if not results:
            return "未找到相关搜索结果。"

        formatted_results = []
        for i, r in enumerate(results.get("results", []), 1):
            formatted_results.append(
                f"[{i}] {r.get('title', '无标题')}\n"
                f"    来源: {r.get('url', '未知')}\n"
                f"    摘要: {r.get('content', '无内容')[:200]}..."
            )

        return "搜索结果:\n\n" + "\n\n".join(formatted_results)

    except Exception as e:
        return f"搜索失败: {str(e)}"


def get_tavily_tools() -> List[BaseTool]:
    """获取 Tavily 工具列表"""
    return [tavily_search_tool]
