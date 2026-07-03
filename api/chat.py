from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from config.database import SessionLocal, get_db
from schemas.chat import ConversationCreate, ConversationResponse, ChatMessageCreate, ChatMessageResponse, ChatRequest, ChatStreamRequest
from crud import chat as chat_crud
from config.security import get_current_user, TokenData
from providers.factory import get_llm_provider
from services.utils import generate_conversation_title
import json

router = APIRouter(prefix="/chat", tags=["聊天"])


@router.post("/conversation/create", response_model=ConversationResponse)
def create_conversation(
    data: ConversationCreate,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """创建新对话"""
    return chat_crud.create_conversation(db, current_user.user_id, data.title)


@router.get("/conversation/list", response_model=list[ConversationResponse])
def get_conversations(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户的所有对话列表"""
    return chat_crud.get_user_conversations(db, current_user.user_id)


@router.post("/message/save")
def save_message(
    message: ChatMessageCreate,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """保存聊天消息"""
    conversation = chat_crud.get_conversation_by_id(db, message.conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    chat_crud.create_message(db, message.conversation_id, message.role, message.content)
    chat_crud.update_conversation_time(db, message.conversation_id)
    return {"msg": "保存成功！"}


@router.get("/conversation/{conversation_id}/history", response_model=list[ChatMessageResponse])
def get_history(
    conversation_id: int,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取某个对话的历史消息"""
    conversation = chat_crud.get_conversation_by_id(db, conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    messages = chat_crud.get_conversation_messages(db, conversation_id)
    return [ChatMessageResponse(role=m.role, content=m.content) for m in messages]


@router.post("/chat")
def chat(
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """聊天接口：发送消息，获取AI回复"""
    conversation = chat_crud.get_conversation_by_id(db, request.conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 获取对话历史，构建消息列表（限制最近10条）
    history = chat_crud.get_conversation_messages(db, request.conversation_id, limit=10)
    messages = [{"role": "system", "content": request.system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.content})

    # 调用大模型
    llm = get_llm_provider()
    ai_response = llm.chat(messages)

    # 保存用户消息和AI回复
    chat_crud.create_message(db, request.conversation_id, "user", request.content)
    chat_crud.create_message(db, request.conversation_id, "assistant", ai_response)
    chat_crud.update_conversation_time(db, request.conversation_id)

    return {"role": "assistant", "content": ai_response}


async def generate_stream_response(db, conversation_id, content, system_prompt):
    """生成流式响应的生成器"""
    # 获取对话历史，构建消息列表
    history = chat_crud.get_conversation_messages(db, conversation_id, limit=10)
    
    # 检查是否是对话的第一条消息（需要生成标题）
    is_first_message = len(history) == 0
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": content})

    # 调用大模型流式接口
    llm = get_llm_provider()
    
    full_response = ""
    for chunk in llm.chat_stream(messages):
        full_response += chunk
        # SSE 格式：data: {"content": "xxx"}\n\n
        yield f"data: {json.dumps({'content': chunk})}\n\n"
    
    # 保存完整消息到数据库
    chat_crud.create_message(db, conversation_id, "user", content)
    chat_crud.create_message(db, conversation_id, "assistant", full_response)
    chat_crud.update_conversation_time(db, conversation_id)
    
    # 如果是第一条消息，自动生成标题
    if is_first_message:
        title = await generate_conversation_title(content)
        chat_crud.update_conversation_title(db, conversation_id, title)
        yield f"data: {json.dumps({'title': title})}\n\n"
    
    yield "data: [DONE]\n\n"


@router.post("/stream")
def chat_stream(
    request: ChatStreamRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """流式聊天接口：SSE 逐字返回AI回复"""
    conversation = chat_crud.get_conversation_by_id(db, request.conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")

    return StreamingResponse(
        generate_stream_response(db, request.conversation_id, request.content, request.system_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
