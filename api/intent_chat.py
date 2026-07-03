from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from config.database import get_db
from config.security import get_current_user, TokenData
from services.intent_recognition import get_intent_recognizer, intent_router
from providers.factory import get_llm_provider
from rag.service import get_rag_service
from crud.chat import create_message, get_conversation_by_id, get_conversation_messages, update_conversation_time, update_conversation_title
from services.utils import generate_conversation_title
from schemas.chat import IntentChatRequest
from fastapi.responses import StreamingResponse
import json

router = APIRouter(prefix="/chat", tags=["Intent Chat"])


# 注册意图处理器
@intent_router.register("rag_query")
def handle_rag_query(intent_result, user_message, **kwargs):
    """处理RAG查询意图 - 从本地文件加载文档"""
    rag_service = get_rag_service()
    
    # 检查是否有加载的文档
    if rag_service.get_documents_count() == 0:
        # 如果没有文档，降级为普通聊天
        llm = get_llm_provider()
        response = llm.chat([{"role": "user", "content": user_message}])
        return {
            "intent": "rag_query",
            "response": response,
            "use_rag": False,
            "entities": intent_result.get("entities", []),
            "confidence": intent_result.get("confidence", 0)
        }
    
    # 使用本地文档进行 RAG 查询
    response = rag_service.chat_with_context(user_message)
    
    return {
        "intent": "rag_query",
        "response": response,
        "use_rag": True,
        "entities": intent_result.get("entities", []),
        "confidence": intent_result.get("confidence", 0)
    }


@intent_router.register("chat")
def handle_chat(intent_result, user_message, **kwargs):
    """处理普通聊天意图"""
    llm = get_llm_provider()
    messages = [{"role": "user", "content": user_message}]
    response = llm.chat(messages)
    
    return {
        "intent": "chat",
        "response": response,
        "use_rag": False,
        "entities": intent_result.get("entities", []),
        "confidence": intent_result.get("confidence", 0)
    }


@intent_router.register("unknown")
def handle_unknown(intent_result, user_message, **kwargs):
    """处理未知意图"""
    llm = get_llm_provider()
    messages = [{"role": "user", "content": user_message}]
    response = llm.chat(messages)
    
    return {
        "intent": "unknown",
        "response": response,
        "use_rag": False,
        "entities": intent_result.get("entities", []),
        "confidence": intent_result.get("confidence", 0)
    }


@router.post("/intent-chat")
def intent_based_chat(
    message: str,
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """
    基于意图识别的统一聊天接口
    
    根据用户输入自动识别意图，并路由到相应的处理逻辑：
    - rag_query: 使用RAG查询文档
    - chat: 普通聊天
    - command: 命令执行
    - unknown: 无法识别的意图
    """
    # 验证对话存在且属于当前用户
    conversation = get_conversation_by_id(db, conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 1. 意图识别
    intent_recognizer = get_intent_recognizer()
    intent_result = intent_recognizer.recognize_intent(message)
    
    # 2. 根据意图路由处理
    result = intent_router.route(
        intent_result,
        db=db,
        user_message=message,
        conversation_id=conversation_id
    )
    
    # 3. 保存消息到数据库
    create_message(db, conversation_id, "user", message)
    create_message(db, conversation_id, "assistant", result.get("response", ""))
    
    return result


@router.post("/intent-chat/stream")
def intent_based_chat_stream(
    request: IntentChatRequest,
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """
    基于意图识别的流式聊天接口
    """
    message = request.content
    conversation_id = request.conversation_id
    
    # 验证对话存在且属于当前用户
    conversation = get_conversation_by_id(db, conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 判断是否是第一条消息
    is_first_message = get_conversation_messages(db, conversation_id, limit=1) == []
    # print(is_first_message)
    
    # 保存用户消息
    create_message(db, conversation_id, "user", message)   

    
    # 意图识别
    intent_recognizer = get_intent_recognizer()
    intent_result = intent_recognizer.recognize_intent(message)
    print(json.dumps(intent_result, indent=2, ensure_ascii=False))
    
    async def generate():
        intent = intent_result.get("intent", "unknown")
        full_response = ""  # 累积完整响应
        
        # 根据意图选择流式处理
        if intent == "rag_query":
            rag_service = get_rag_service()
            if rag_service.is_available():
                yield f"data: {json.dumps({'intent': 'rag_query', 'use_rag': True})}\n\n"
                for chunk in rag_service.chat_stream_with_context(message):
                    if chunk == "[DONE]":
                        yield "data: [DONE]\n\n"
                    else:
                        data = json.loads(chunk)
                        if "content" in data:
                            full_response += data["content"]
                        yield f"data: {chunk}\n\n"
            else:
                # 构建错误信息
                error_info = {}
                if rag_service.load_error:
                    error_info["error"] = f"文档加载失败: {rag_service.load_error}"
                yield f"data: {json.dumps({'intent': 'rag_query', 'use_rag': False, **error_info})}\n\n"
                llm = get_llm_provider()
                for chunk in llm.chat_stream([{"role": "user", "content": message}]):
                    full_response += chunk
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
        else:
            # 普通聊天或其他意图
            llm = get_llm_provider()
            yield f"data: {json.dumps({'intent': intent, 'use_rag': False})}\n\n"
            for chunk in llm.chat_stream([{"role": "user", "content": message}]):
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        
        # 流式响应完成后保存助手消息和更新对话时间
        create_message(db, conversation_id, "assistant", full_response)
        update_conversation_time(db, conversation_id)
        
        # 如果是第一条消息，自动生成标题
        if is_first_message:
            title = await generate_conversation_title(message)
            update_conversation_title(db, conversation_id, title)
            yield f"data: {json.dumps({'type': 'title', 'title': title})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/intent-test")
def test_intent_recognition(
    message: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    测试意图识别接口（仅返回意图识别结果，不执行实际对话）
    """
    intent_recognizer = get_intent_recognizer()
    result = intent_recognizer.recognize_intent(message)
    return result
