from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from config.database import get_db
from config.security import get_current_user, TokenData
from schemas.chat import IntentChatRequest
from crud.chat import create_message, get_conversation_by_id, get_conversation_messages, update_conversation_time, update_conversation_title
from services.utils import generate_conversation_title
from services.agent import chat_unified, stream_agent_response, stream_agent_logic
from services.task_graph import stream_task_graph
import json
import re

router = APIRouter(prefix="/chat", tags=["Agent Chat"])


@router.post("/agent")
async def agent_chat(
    request: IntentChatRequest,
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Agent 统一聊天接口（非流式）

    自动调度 RAG 知识库、Tavily 搜索、直接回答。
    """
    conversation = get_conversation_by_id(db, request.conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 使用 conversation_id 作为 thread_id，使 Agent 能从 PostgreSQL 加载历史上下文
    response = await chat_unified(request.content, thread_id=str(request.conversation_id))

    create_message(db, request.conversation_id, "user", request.content)
    create_message(db, request.conversation_id, "assistant", response)
    update_conversation_time(db, request.conversation_id)

    return {"role": "assistant", "content": response}


@router.post("/agent/stream")
def agent_chat_stream(
    request: IntentChatRequest,
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Agent 统一聊天接口（流式 SSE）

    自动调度 RAG 知识库、Tavily 搜索、直接回答，逐块返回。
    """
    conversation = get_conversation_by_id(db, request.conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")

    is_first_message = get_conversation_messages(db, request.conversation_id, limit=1) == []

    create_message(db, request.conversation_id, "user", request.content)

    async def generate():
        full_response = ""

        async for payload in stream_agent_logic(request.content, thread_id=str(request.conversation_id)):
        # for payload in stream_agent_response(request.content):
            payload_type = payload.get("type")

            if payload_type == "content":
                # 1. 累积文本内容（用于后续可能的日志记录或持久化）
                full_response += payload.get("content", "")
                # 2. 原样透传给前端
                final_payload = payload

            elif payload_type == "status":
                # 3. 状态信息转换为 tool 类型发给前端
                final_payload = {
                    "type": "tool",
                    "tool_name": payload.get("tool_name", "未知工具"),
                    "content": payload.get("content", ""),
                    "intent": payload.get("intent", ""),
                    "sources": payload.get("sources", "")
                }
            elif payload_type == "sources":
                final_payload = {
                    "type": "sources",
                    "sources": payload.get("sources", "")
                }            
            else:
                continue

            # 4. 统一在这里做 JSON 序列化，保证绝对安全
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
      

        # 保存助手回复
        if full_response:
            create_message(db, request.conversation_id, "assistant", full_response)
        update_conversation_time(db, request.conversation_id)

        # 第一条消息自动生成标题
        if is_first_message:
            title = await generate_conversation_title(request.content)
            update_conversation_title(db, request.conversation_id, title)
            yield f"data: {json.dumps({'type': 'title', 'title': title})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/task-stream")
def task_chat_stream(
    request: IntentChatRequest,
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """
    任务编排流式接口（意图引导 + 多步规划 + 执行）

    流程：意图分类 → (不明确时)追问 → 任务规划 → 逐步执行 → 汇总
    多轮引导与执行状态通过 PostgreSQL Checkpointer 按 conversation_id 持久化。

    SSE 事件类型：
    - intent:  意图识别结果
    - clarify: 需要追问（本轮结束，等待用户下轮回答）
    - plan:    任务规划步骤列表
    - status:  某步骤执行完毕
    - content: 最终回答
    - title:   首条消息自动生成的对话标题
    """
    conversation = get_conversation_by_id(db, request.conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")

    is_first_message = get_conversation_messages(db, request.conversation_id, limit=1) == []
    create_message(db, request.conversation_id, "user", request.content)

    async def generate():
        full_response = ""

        async for payload in stream_task_graph(
            request.content,
            thread_id=str(request.conversation_id)
        ):
            # 追问：整句覆盖；最终回答：分块累加（finalize 流式输出多个 content chunk）
            if payload.get("type") == "clarify":
                full_response = payload.get("content", "")
            elif payload.get("type") == "content":
                full_response += payload.get("content", "")

            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 保存助手回复（追问或最终回答）
        if full_response:
            create_message(db, request.conversation_id, "assistant", full_response)
        update_conversation_time(db, request.conversation_id)

        # 首条消息自动生成标题
        if is_first_message:
            title = await generate_conversation_title(request.content)
            update_conversation_title(db, request.conversation_id, title)
            yield f"data: {json.dumps({'type': 'title', 'title': title})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

