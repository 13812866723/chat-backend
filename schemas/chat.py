from pydantic import BaseModel
from datetime import datetime


class ConversationCreate(BaseModel):
    title: str = "新对话"


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    conversation_id: int
    role: str
    content: str


class ChatMessageResponse(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    conversation_id: int
    content: str
    system_prompt: str = "你是一个有用的AI助手。"


class ChatStreamRequest(BaseModel):
    conversation_id: int
    content: str
    system_prompt: str = "你是一个有用的AI助手。"


class IntentChatRequest(BaseModel):
    """意图聊天请求模型"""
    content: str
    conversation_id: int
