"""Agent 模块测试"""
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_get_unified_tools():
    """测试获取统一工具列表"""
    from services.agent import get_unified_tools

    tools = get_unified_tools()
    print("=== 统一工具列表测试 ===")
    print(f"获取到 {len(tools)} 个工具:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:60]}...")
    print()


def test_create_unified_agent():
    """测试创建统一 Agent"""
    from services.agent import create_unified_agent

    print("=== 创建统一 Agent 测试 ===")
    try:
        agent = create_unified_agent()
        print(f"Agent 类型: {type(agent).__name__}")
        print(f"Agent 对象: {agent}")
        # 安全地检查属性
        if hasattr(agent, 'tools'):
            print(f"绑定工具数: {len(agent.tools)}")
        elif hasattr(agent, 'get_tools'):
            print(f"绑定工具数: {len(agent.get_tools())}")
        else:
            print(f"Agent 属性: {[a for a in dir(agent) if not a.startswith('_')]}")
    except Exception as e:
        print(f"创建 Agent 失败: {type(e).__name__}: {e}")
    print()


def test_chat_unified():
    """测试统一 Agent 聊天（直接回答，不调用工具）"""
    from services.agent import chat_unified

    query = "1+1等于多少？"
    print(f"=== 统一 Agent 聊天测试 ===")
    print(f"用户问题: {query}")
    result = chat_unified(query)
    print(f"Agent 回答: {result}")
    print()


def test_chat_unified_with_rag():
    """测试统一 Agent 聊天（触发 RAG 工具）"""
    from services.agent import chat_unified

    query = "民法典合同编中关于违约责任的规定是什么？"
    print(f"=== 统一 Agent RAG 测试 ===")
    print(f"用户问题: {query}")
    result = chat_unified(query)
    print(f"Agent 回答: {result}")
    print()


def test_chat_unified_with_search():
    """测试统一 Agent 聊天（触发 Tavily 搜索）"""
    from services.agent import chat_unified

    query = "今天有什么重大新闻？"
    print(f"=== 统一 Agent 搜索测试 ===")
    print(f"用户问题: {query}")
    result = chat_unified(query)
    print(f"Agent 回答: {result}")
    print()

from services.agent import stream_agent_response, create_unified_agent

async def test():

    # async for chunk in stream_agent_response(agent, "今天天气怎么样？"):
    # async for chunk in stream_agent_response( "民法典合同编中关于违约责任的规定是什么？"):
    # async for chunk in stream_agent_response( "你好"):
    async for chunk in stream_agent_response_debug("今天有什么新闻吗"):
        print(chunk, end="") # end="" 防止换行，保持流式效果

async def stream_agent_response_debug(user_input: str):
    agent = create_unified_agent()
    inputs = {"messages": [("user", user_input)]}
    
    output_file = os.path.join(os.path.dirname(__file__), "debug_output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        # 使用 stream_mode="messages"
        async for chunk in agent.astream(inputs, stream_mode="messages"):
            line = f"=== 原始 chunk ===\n{chunk}\n"
            f.write(line)
            print(line)
            yield "data: [DEBUG]\n\n"
        
        f.write("=== DONE ===\n")
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    # test_get_unified_tools()
    # test_create_unified_agent()
    # test_chat_unified()
    # test_chat_unified_with_rag()
    # test_chat_unified_with_search()
    import asyncio  

    # 运行测试
    asyncio.run(test())
