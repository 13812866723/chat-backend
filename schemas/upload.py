from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    success: bool
    message: str
    file_id: str
    filename: str
    document_count: int
    chunk_count: int
    content_preview: str = ""


class AnalyzeRequest(BaseModel):
    conversation_id: int
    content: str
    file_ids: list[str] | None = None
