"""Tavily 搜索工具测试"""
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# 设置测试用的 API Key（从环境变量读取）
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")



# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def test_get_tavily_tools():
    """测试获取工具列表"""
    from tools.tavily import get_tavily_tools

    tools = get_tavily_tools()
    print("=== 工具列表测试 ===")
    print(f"获取到 {len(tools)} 个工具:")
    for t in tools:
        print(f"  - {t.name}")
    print()


def test_tavily_search():
    """测试基础搜索功能"""
    from tools.tavily import tavily_search_tool

    result = tavily_search_tool.invoke({"query": "刘德华生日", "max_results": 3})
    print("=== 基础搜索测试 ===")
    print(result)
    print()


if __name__ == "__main__":
    test_get_tavily_tools()
    test_tavily_search()
