# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from config.database import Base


def utc_now():
    return datetime.now(timezone.utc)


# 1. 用户表
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)  # 用户名
    hashed_password = Column(String(255))                   # 密码哈希
    created_at = Column(DateTime, default=utc_now)

    # 关系：一个用户有多个对话
    conversations = relationship("Conversation", back_populates="user")


# 2. 对话表
class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))      # 外键：属于哪个用户
    title = Column(String(100), default="新对话")          # 对话标题
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # 关系：一个对话有多条消息
    messages = relationship("ChatMessage", back_populates="conversation")
    # 关系：一个对话属于一个用户
    user = relationship("User", back_populates="conversations")


# 3. 聊天记录表
class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))  # 外键：属于哪个对话
    role = Column(String(20))                                           # 角色：'user' 或 'assistant'
    content = Column(Text)                                              # 聊天内容
    created_at = Column(DateTime, default=utc_now)

    # 关系：一条消息属于一个对话
    conversation = relationship("Conversation", back_populates="messages")