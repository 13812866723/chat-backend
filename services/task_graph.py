"""任务编排图 - 意图引导 + 多步规划 + 执行

基于 LangGraph StateGraph 实现多轮任务路由：

    START
      │
      ▼
   entry_route ──(上轮在等追问回答)──► plan
      │
      │ (新输入)
      ▼
   classify_intent
      │
      ▼
   route_after_classify ──(置信度低/未知)──► clarify ──► END
      │
      │ (意图清晰)
      ▼
   plan ──► execute_step ──► route_after_execute
                                 │
                                 ├─(还有步骤)─► execute_step (循环)
                                 │
                                 └─(全部完成)─► finalize ──► END

状态由 PostgreSQL Checkpointer 按 thread_id 持久化，
多轮引导（追问→用户回答→继续）天然支持：
- 第 1 轮命中 clarify → 返回追问问题 → END
- 第 2 轮用户回答 → entry_route 检测到 needs_clarification → 直接 plan
  （复用上轮意图 + 合并原始问题与补充回答）
"""
import asyncio
from typing import TypedDict, List, Dict, Any, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage, HumanMessage

from providers.factory import get_llm_provider
from services.intent_recognition import get_intent_recognizer


# ========== 状态定义 ==========
class TaskState(TypedDict):
    messages: Annotated[list, add_messages]   # 累积对话消息
    raw_intent: Dict[str, Any]                # 初步意图识别结果
    needs_clarification: bool                 # 是否在等待用户追问回答
    clarification_q: str                      # 上轮发出的追问问题
    original_query: str                       # 触发追问的原始问题（用于第 2 轮合并）
    plan_steps: List[str]                     # 拆解出的执行步骤
    current_step: int                         # 当前执行步骤索引
    step_results: List[str]                   # 各步骤执行结果
    last_step: str                            # 最近一次执行的步骤描述
    final_answer: str                         # 最终答案


# 触发追问的置信度阈值
CLARIFY_CONFIDENCE_THRESHOLD = 0.6


# ========== 辅助函数 ==========
def _last_user_content(state: TaskState) -> str:
    """从 state.messages 中取出最后一条用户消息文本"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


async def _llm_chat(system: str, user: str) -> str:
    """异步调用 LLM（同步 provider 用 to_thread 包裹，避免阻塞事件循环）"""
    provider = get_llm_provider()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await asyncio.to_thread(provider.chat, messages)


async def _llm_chat_stream(system: str, user: str):
    """异步流式调用 LLM，逐 chunk 产出文本。

    provider.chat_stream 是同步生成器（基于 requests stream），
    通过 run_in_executor 逐次拉取 next，避免阻塞事件循环。
    """
    provider = get_llm_provider()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    gen = provider.chat_stream(messages)
    loop = asyncio.get_running_loop()
    sentinel = object()
    while True:
        chunk = await loop.run_in_executor(None, next, gen, sentinel)
        if chunk is sentinel:
            break
        yield chunk


# ========== 节点实现 ==========
async def classify_intent(state: TaskState) -> dict:
    """意图分类节点：复用 IntentRecognizer 识别意图与置信度"""
    user_msg = _last_user_content(state)
    recognizer = get_intent_recognizer()
    # IntentRecognizer.recognize_intent 是同步方法，放到线程池执行
    result = await asyncio.to_thread(recognizer.recognize_intent, user_msg)
    return {"raw_intent": result}


async def clarify(state: TaskState) -> dict:
    """引导追问节点：意图不明确时生成追问问题，并记下原始问题待下轮合并"""
    intent = state.get("raw_intent", {})
    user_msg = _last_user_content(state)
    reasoning = intent.get("reasoning", "")

    sys = (
        "你是一个对话引导助手。用户刚才的输入意图不够明确，"
        "请生成一个简短、友好的追问，帮助确认用户真正想做什么。"
        "只输出追问本身，不要多余解释。"
    )
    prompt = (
        f"用户输入：{user_msg}\n"
        f"初步意图判断：{intent.get('intent')}（{reasoning}）\n"
        f"请生成追问："
    )
    question = (await _llm_chat(sys, prompt)).strip()

    return {
        "needs_clarification": True,
        "clarification_q": question,
        "original_query": user_msg,
        "messages": [AIMessage(content=question)],
    }


async def plan(state: TaskState) -> dict:
    """任务规划节点：根据意图把任务拆解为 1-3 个可执行步骤

    步骤格式（前缀:参数）：
    - rag:<查询内容>     查询本地知识库
    - search:<查询内容>  联网搜索
    - answer:<要点>      直接用 LLM 回答
    """
    intent = state.get("raw_intent", {})
    intent_type = intent.get("intent", "chat")

    # 若是追问后的第二轮，合并原始问题与用户补充回答
    if state.get("needs_clarification") and state.get("original_query"):
        effective_query = f"{state['original_query']}（用户补充：{_last_user_content(state)}）"
    else:
        effective_query = _last_user_content(state)

    sys = (
        "你是一个任务规划器。根据用户意图和输入，把任务拆解为 1-3 个明确的执行步骤。\n"
        "步骤格式说明：\n"
        "- 'rag:<查询内容>' 表示需要查询本地知识库\n"
        "- 'search:<查询内容>' 表示需要联网搜索\n"
        "- 'answer:<要点>' 表示直接用 LLM 回答\n\n"
        "要求：\n"
        "- 只输出步骤列表，每行一个步骤，不要编号和多余文字\n"
        "- 简单问题只给 1 个步骤\n"
        "- 复杂问题最多 3 个步骤"
    )
    prompt = f"用户意图：{intent_type}\n用户输入：{effective_query}\n\n请输出执行步骤："
    raw = await _llm_chat(sys, prompt)

    steps = [s.strip() for s in raw.strip().splitlines() if s.strip()]
    if not steps:
        steps = [f"answer:{effective_query}"]

    return {
        "plan_steps": steps,
        "current_step": 0,
        "step_results": [],
        "last_step": "",
        "needs_clarification": False,   # 进入规划后清除追问标记
        "original_query": "",           # 清空，避免污染后续轮次
    }


async def execute_step(state: TaskState) -> dict:
    """执行单个步骤：按步骤前缀调用对应工具/LLM"""
    step_idx = state["current_step"]
    step = state["plan_steps"][step_idx]
    prefix, _, arg = step.partition(":")
    prefix = prefix.strip().lower()
    arg = arg.strip() or _last_user_content(state)

    if prefix == "rag":
        from rag.service import rag_chat_tool
        result = await asyncio.to_thread(rag_chat_tool.invoke, arg)
    elif prefix == "search":
        from tools.tavily import tavily_search_tool
        result = await asyncio.to_thread(tavily_search_tool.invoke, arg)
    else:  # answer 或其它前缀统一走 LLM
        result = await _llm_chat("你是一个智能助手，请直接回答用户问题。", arg)

    new_results = list(state.get("step_results", []))
    new_results.append(result)

    return {
        "step_results": new_results,
        "current_step": step_idx + 1,
        "last_step": step,
    }


async def finalize(state: TaskState) -> dict:
    """汇总节点：综合各步骤结果生成最终回答，并流式推送 token。

    通过 LangGraph 的 custom stream writer 把 LLM 生成的 token 逐个
    推送到外层流（stream_mode="custom"），实现逐字输出。
    单步骤场景下结果已由工具/LLM 产生，直接作为单个 chunk 推送。
    """
    writer = get_stream_writer()
    user_msg = _last_user_content(state)
    results = state.get("step_results", [])

    # 单步骤：结果已完整，直接推送（工具返回的是完整字符串，无法再流式）
    if len(results) <= 1:
        final = results[0] if results else ""
        if final:
            writer({"type": "content", "content": final})
    else:
        # 多步骤：流式综合，逐 token 推送
        sys = (
            "你是一个智能助手。以下是针对用户问题分步收集到的信息，"
            "请综合这些信息生成一个连贯、完整的最终回答。"
            "不要逐条罗列步骤结果，要自然地组织语言。"
        )
        context = "\n\n".join(f"[步骤{i+1}结果]\n{r}" for i, r in enumerate(results))
        prompt = f"用户原始问题：{user_msg}\n\n{context}\n\n请生成最终回答："

        chunks = []
        async for chunk in _llm_chat_stream(sys, prompt):
            chunks.append(chunk)
            writer({"type": "content", "content": chunk})
        final = "".join(chunks)

    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }


# ========== 路由函数 ==========
def route_entry(state: TaskState) -> str:
    """入口路由：上轮在等追问回答则直接进入规划，否则先分类"""
    if state.get("needs_clarification"):
        return "plan"
    return "classify_intent"


def route_after_classify(state: TaskState) -> str:
    """分类后路由：置信度低或未知意图则追问，否则规划"""
    intent = state.get("raw_intent", {})
    confidence = intent.get("confidence", 0)
    intent_type = intent.get("intent", "unknown")
    if intent_type == "unknown" or confidence < CLARIFY_CONFIDENCE_THRESHOLD:
        return "clarify"
    return "plan"


def route_after_execute(state: TaskState) -> str:
    """执行后路由：还有步骤则继续，否则汇总"""
    if state["current_step"] < len(state["plan_steps"]):
        return "execute_step"
    return "finalize"


# ========== 构建图 ==========
def build_task_graph(checkpointer=None):
    """构建任务编排 StateGraph 并编译"""
    g = StateGraph(TaskState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("clarify", clarify)
    g.add_node("plan", plan)
    g.add_node("execute_step", execute_step)
    g.add_node("finalize", finalize)

    g.add_conditional_edges(START, route_entry, {
        "plan": "plan",
        "classify_intent": "classify_intent",
    })
    g.add_conditional_edges("classify_intent", route_after_classify, {
        "clarify": "clarify",
        "plan": "plan",
    })
    g.add_edge("clarify", END)
    g.add_edge("plan", "execute_step")
    g.add_conditional_edges("execute_step", route_after_execute, {
        "execute_step": "execute_step",
        "finalize": "finalize",
    })
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


# ========== 流式输出 ==========
async def stream_task_graph(user_input: str, thread_id: Optional[str] = None):
    """
    运行任务图并产出结构化事件（供 SSE 推送）。

    事件类型：
    - {"type":"intent","intent":...,"confidence":...}  意图识别结果
    - {"type":"clarify","content":...}                  需要追问（本轮到此结束）
    - {"type":"plan","steps":[...]}                     任务规划步骤
    - {"type":"status","step":...,"content":...}        某步骤执行完毕
    - {"type":"content","content":...}                  最终回答（多步骤时逐 token 流式）
    """
    from services.agent import get_checkpointer

    checkpointer = get_checkpointer() if thread_id else None
    graph = build_task_graph(checkpointer=checkpointer)

    inputs = {"messages": [HumanMessage(content=user_input)]}
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    # 双流模式：updates 拿结构化状态事件，custom 拿 finalize 的流式 token
    async for mode, chunk in graph.astream(
        inputs, stream_mode=["updates", "custom"], config=config
    ):
        if mode == "custom":
            # finalize 通过 writer 推送的 token chunk，直接透传
            yield chunk
            continue

        # mode == "updates"
        for node_name, update in chunk.items():
            if node_name == "classify_intent":
                intent = update.get("raw_intent", {})
                yield {
                    "type": "intent",
                    "intent": intent.get("intent"),
                    "confidence": intent.get("confidence"),
                    "reasoning": intent.get("reasoning", ""),
                }

            elif node_name == "clarify":
                yield {"type": "clarify", "content": update.get("clarification_q", "")}

            elif node_name == "plan":
                yield {"type": "plan", "steps": update.get("plan_steps", [])}

            elif node_name == "execute_step":
                # current_step 已自增，刚执行的是 current_step-1
                done_idx = update.get("current_step", 0) - 1
                yield {
                    "type": "status",
                    "step": update.get("last_step", ""),
                    "content": f"步骤 {done_idx + 1} 执行完毕",
                }

            # finalize 的最终回答已通过 custom 流推送，此处不再重复 yield
