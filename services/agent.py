"""Agent 模块 - 统一调度 RAG 和 Tavily 搜索工具"""
from typing import Union, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from providers.factory import get_llm_provider
from config.database import DATABASE_URL
import json

def get_unified_tools() -> List[BaseTool]:
    """
    获取统一工具列表（RAG + Tavily 搜索）。

    RAG 工具用于回答本地知识库相关问题，
    Tavily 工具用于获取实时网络信息。
    """
    from rag.service import rag_chat_tool
    from tools.tavily import get_tavily_tools

    tools = [rag_chat_tool] + get_tavily_tools()
    return tools


# ========== PostgreSQL Checkpointer（短期记忆持久化）==========
# 模块级单例：复用同一个数据库连接的 checkpointer，避免每次调用都重建连接
# 注意：使用 AsyncPostgresSaver（而非同步 PostgresSaver），因为流式接口
# agent.astream() 内部会调用 checkpointer.aget_tuple 等异步方法，
# 同步版本的对应方法未实现（会抛 NotImplementedError）。
_async_checkpointer_instance = None


async def init_checkpointer():
    """
    初始化异步 PostgreSQL Checkpointer 单例（应在应用启动时调用一次）。

    建立 psycopg 异步连接并执行 setup() 创建检查点表结构。
    初始化完成后，get_checkpointer() 即可同步返回该单例，
    使 create_unified_agent 等同步构造函数能直接复用。
    """
    global _async_checkpointer_instance
    if _async_checkpointer_instance is None:
        import psycopg
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # autocommit=True 与 row_factory=dict_row 是 PostgresSaver 的硬性要求：
        # - autocommit：保证 setup() 建表能正确提交
        # - dict_row：checkpointer 内部按列名访问行数据，默认 tuple_row 会报 TypeError
        conn = await psycopg.AsyncConnection.connect(
            DATABASE_URL,
            autocommit=True,
            row_factory=dict_row,
        )
        _async_checkpointer_instance = AsyncPostgresSaver(conn)
        # 首次使用时创建 checkpoints / checkpoint_writes / checkpoint_blobs 等表
        await _async_checkpointer_instance.setup()
    return _async_checkpointer_instance


def get_checkpointer():
    """
    同步返回已初始化的异步 PostgreSQL Checkpointer 单例。

    必须先在应用启动时调用 await init_checkpointer() 完成初始化，
    否则此处会抛出 RuntimeError。设计为同步以便在 create_unified_agent
    等同步构造函数中直接使用，把"建连接/建表"这件异步事和"构造 agent"解耦。
    """
    if _async_checkpointer_instance is None:
        raise RuntimeError(
            "Checkpointer 尚未初始化，请先在应用启动时调用 await init_checkpointer()"
        )
    return _async_checkpointer_instance


def create_unified_agent(
    llm: Union[str, BaseChatModel, None] = None,
    tools: Union[list[BaseTool], None] = None,
    system_message: str = (
        "你是一个智能助手，具备以下能力：\n"
        "1. 本地知识库问答（包含中华人民共和国民法典合同编）\n"
        "2. 实时网络搜索能力\n"
        "3. 通用知识问答能力\n\n"
        "【工作流程：意图判断 → 流程规划 → 执行】\n"
        "面对用户输入，请按以下步骤思考后再行动：\n"
        "1. 意图判断：先理解用户到底想做什么。是查询本地知识库？需要联网获取实时信息？还是可直接回答的常识/计算/推理？若输入过于模糊（指代不明、动词无宾语、多意图混杂），先反问澄清，不要贸然调用工具。\n"
        "2. 流程规划：明确意图后，判断需要几步完成。简单问题直接回答；复杂问题可拆为多步（如先查知识库、再联网补充），按顺序依次调用工具。\n"
        "3. 执行：按规划调用工具或直接作答，工具返回结果后综合分析再决定是否进入下一步。\n\n"
        "【工具使用原则】\n"
        "- 当问题涉及合同编法条、民事法律问题、文档内容等知识库相关问题时，使用 rag_chat_tool\n"
        "- 当需要获取最新信息、实时新闻、天气预报、股价汇率等网络实时信息时，使用 tavily_search_tool\n"
        "- 对于常识性问题、简单计算、逻辑推理等不需要外部信息的问题，直接回答，不要调用任何工具\n"
        "- 只有在必要时才使用工具，避免不必要的工具调用\n"
        "- 可以同时使用多个工具来综合回答复杂问题\n\n"
        "【重要提示】\n"
        "- 优先评估是否可以直接回答问题，只有在确实需要外部信息或知识库内容时才调用工具\n"
        "- 意图不明确时，宁可先追问一句，也不要乱调工具或乱答\n"
        "- 多步任务请逐步执行，每步基于上一步结果再决策，不要一次性盲目调用所有工具"
    ),
    use_memory: bool = True,
) -> BaseChatModel:
    """
    创建统一 Agent（RAG + Tavily 搜索 + 直接回答）。

    该 Agent 同时绑定本地知识库检索工具和在线搜索工具，
    可根据问题类型自动选择合适的工具或直接回答。

    参数:
        llm: 大模型服务，传入 None 则自动从配置获取
        tools: 工具列表，传入 None 则使用默认的统一工具（RAG + Tavily）
        system_message: 系统提示词
        use_memory: 是否启用 PostgreSQL 短期记忆（Checkpointer）。
            启用后，需在调用时通过 config 传入 thread_id 以实现多轮记忆。

    返回:
        绑定了多个工具的 LLM Agent
    """
    from langchain.agents import create_agent

    # 获取 LLM
    if llm is None:
        llm_service = get_llm_provider()
        if hasattr(llm_service, 'chat_model'):
            llm = llm_service.chat_model
        else:
            llm = llm_service

    # 获取工具
    if tools is None:
        tools = get_unified_tools()

    # 获取 Checkpointer（短期记忆持久化到 PostgreSQL，需已在 startup 初始化）
    checkpointer = get_checkpointer() if use_memory else None

    # 创建 Agent
    agent = create_agent(
        model=llm,              # 第一个参数通常可以是位置参数，但为了清晰也建议写明
        tools=tools,            # 明确指定 'tools' 参数
        system_prompt=system_message, # 明确指定 'system_prompt' 参数
        checkpointer=checkpointer,    # 传入 Checkpointer 实现短期记忆
    )

    return agent


async def chat_unified(query: str, thread_id: Optional[str] = None) -> str:
    """
    使用统一 Agent 进行聊天（RAG + Tavily）。

    参数:
        query: 用户问题
        thread_id: 会话线程 ID（通常为 conversation_id 的字符串形式）。
            传入相同 thread_id 时，Agent 会从 PostgreSQL 加载该会话的
            历史上下文，实现多轮短期记忆。

    返回:
        Agent 回答
    """
    # 创建统一 Agent 实例
    agent = create_unified_agent()
    # 调用 Agent 并传入用户问题，返回 Agent 的回答
    inputs = {"messages": [{"role": "user", "content": query}]}
    # 通过 thread_id 让 checkpointer 加载/保存该会话的历史状态
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    # 使用 ainvoke（与异步 AsyncPostgresSaver 配套）
    response = await agent.ainvoke(inputs, config=config)
    # response 是 dict，取最后一条消息的内容
    last_message = response["messages"][-1]
    if hasattr(last_message, "content"):
        return last_message.content
    return last_message.get("content", str(last_message))

async def stream_agent_response(user_input: str, thread_id: Optional[str] = None):
    agent = create_unified_agent()
    inputs = {"messages": [("user", user_input)]}
    # 通过 thread_id 让 checkpointer 加载/保存该会话的历史状态
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    async for chunk in agent.astream(inputs, stream_mode="updates", config=config):
        
        # 1. 捕获模型思考与工具调用阶段 ('model' 节点)
        if "model" in chunk:
            message = chunk["model"]["messages"][0]
            
            # 如果大模型决定调用工具
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name", "未知工具")
                    tool_args = tool_call.get("args", {})
                    # 将参数转为易读的字符串
                    args_str = json.dumps(tool_args, ensure_ascii=False) 
                    
                    yield f"data: {{\"type\": \"status\", \"content\": \"🧠 正在调用工具: [{tool_name}]，参数: {args_str}\"}}\n\n"
            
            # 如果大模型直接输出最终文本
            elif message.content:
                yield f"data: {{\"type\": \"content\", \"content\": \"{message.content}\"}}\n\n"

        # 2. 捕获工具执行完毕阶段 ('tools' 节点)
        if "tools" in chunk:
            tool_messages = chunk["tools"]["messages"]
            for tool_msg in tool_messages:
                tool_name = tool_msg.name
                yield f"data: {{\"type\": \"status\", \"content\": \"🔧 工具 [{tool_name}] 执行完毕，正在分析结果...\"}}\n\n"
            
    yield "data: [DONE]\n\n"


async def stream_agent_logic(user_input: str, thread_id: Optional[str] = None):
    """
    业务层：只负责产出结构化数据（字典）
    适配 LangGraph stream_mode="messages"

    参数:
        user_input: 用户输入
        thread_id: 会话线程 ID（通常为 conversation_id 的字符串形式）。
            传入相同 thread_id 时，Agent 会从 PostgreSQL 加载该会话的
            历史上下文，实现多轮短期记忆。
    """
    agent = create_unified_agent()
    inputs = {"messages": [("user", user_input)]}
    # 通过 thread_id 让 checkpointer 加载/保存该会话的历史状态
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    async for chunk in agent.astream(inputs, stream_mode="messages", config=config):
        # 1. 解包元组：(message_chunk, metadata)
        message_chunk, metadata = chunk
        
        # 2. 提取节点名称（从 metadata 中获取，判断当前是哪个节点在执行）
        node_name = metadata.get("langgraph_node", "")
        
        # 3. 处理模型节点的流式输出 (文本和工具调用)
        if node_name == "model":
            # 处理文本增量
            if message_chunk.content:
                yield {"type": "content", "content": message_chunk.content}
            
            # 处理工具调用增量 (tool_call_chunks)
            if hasattr(message_chunk, "tool_call_chunks") and message_chunk.tool_call_chunks:
                for tc_chunk in message_chunk.tool_call_chunks:
                    tool_name = tc_chunk.get("name") 

                    if  tool_name: # 简单判断，只要有 name 就发
                        yield {
                            "type": "status",
                            "tool_name": [tool_name],
                            "intent": [tool_name],
                        }

        # 4. 处理工具节点的执行结果
        elif node_name == "tools":
            # 在 messages 模式下，工具节点输出的通常是一个完整的 ToolMessage
            if hasattr(message_chunk, "name"):
                yield {
                    "type": "sources",
                    "tool_name": message_chunk.name,
                    "sources": message_chunk.content 
                }