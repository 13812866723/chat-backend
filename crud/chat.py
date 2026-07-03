from sqlalchemy.orm import Session
from models import ChatMessage, Conversation
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc)


def create_conversation(db: Session, user_id: int, title: str = "新对话"):
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def get_user_conversations(db: Session, user_id: int):
    return db.query(Conversation)\
        .filter(Conversation.user_id == user_id)\
        .order_by(Conversation.updated_at.desc())\
        .all()


def get_conversation_by_id(db: Session, conversation_id: int):
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()


def update_conversation_time(db: Session, conversation_id: int):
    conversation = get_conversation_by_id(db, conversation_id)
    if conversation:
        conversation.updated_at = utc_now()
        db.commit()


def create_message(db: Session, conversation_id: int, role: str, content: str):
    message = ChatMessage(conversation_id=conversation_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_conversation_messages(db: Session, conversation_id: int, limit: int = None):
    query = db.query(ChatMessage)\
        .filter(ChatMessage.conversation_id == conversation_id)\
        .order_by(ChatMessage.created_at)
    if limit:
        query = query.limit(limit)
    return query.all()


def update_conversation_title(db: Session, conversation_id: int, title: str):
    conversation = get_conversation_by_id(db, conversation_id)
    if conversation:
        conversation.title = title
        db.commit()
