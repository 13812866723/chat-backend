from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from rag.service import get_rag_service
from api.user import get_current_user, TokenData

router = APIRouter(prefix="/rag", tags=["RAG"])


class ReloadResponse(BaseModel):
    success: bool
    message: str
    document_count: int = 0


@router.post("/reload", response_model=ReloadResponse)
def reload_documents(current_user: TokenData = Depends(get_current_user)):
    """重新加载向量库：从本地文件重新加载所有文档"""
    try:
        rag_service = get_rag_service()
        rag_service.reload_documents()
        document_count = rag_service.vectorstore._collection.count()
        return ReloadResponse(
            success=True,
            message="向量库重新加载成功",
            document_count=document_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新加载失败: {str(e)}")


@router.get("/status")
def get_status(current_user: TokenData = Depends(get_current_user)):
    """获取向量库状态"""
    rag_service = get_rag_service()
    document_count = rag_service.vectorstore._collection.count()
    return {
        "available": rag_service.is_available(),
        "document_count": document_count,
        "load_error": rag_service.load_error
    }
