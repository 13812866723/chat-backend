from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from config.database import get_db
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, Docx2txtLoader
from crud.chat import create_message, get_conversation_by_id, get_conversation_messages, update_conversation_time, update_conversation_title
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from providers.factory import get_llm_provider
from services.utils import generate_conversation_title
from schemas.upload import UploadResponse, AnalyzeRequest
from config.security import get_current_user, TokenData
import os
import uuid
import json

router = APIRouter(prefix="/upload", tags=["文件上传"])

# 文件存储目录（进程启动时清理）
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")


def cleanup_upload_dir():
    """启动时清理上传目录"""
    if os.path.exists(UPLOAD_DIR):
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"🗑️ 清理历史文件: {file}")

# 初始化目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
cleanup_upload_dir()


@router.post("/file", response_model=UploadResponse)
async def upload_file(
    current_user: TokenData = Depends(get_current_user),
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """上传文件并存储到服务器"""
    allowed_extensions = {".txt", ".docx", ".doc"}
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}。支持的类型: {', '.join(allowed_extensions)}"
        )
    
    # 生成唯一文件 ID
    file_id = str(uuid.uuid4())
    
    # 保存文件到上传目录
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    try:
        # 加载并分割文档
        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            try:
                loader = UnstructuredWordDocumentLoader(file_path)
            except Exception:
                loader = Docx2txtLoader(file_path)
        
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"\n{'='*60}")
        print(f"📄 文件名: {filename}")
        print(f"� 文件ID: {file_id}")
        print(f"�� 文件大小: {len(content)} bytes")
        print(f"📑 原始文档数量: {len(documents)}")
        print(f"✂️ 分割后 Chunks 数量: {len(chunks)}")
        print(f"📐 分割参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        print(f"💾 保存路径: {file_path}")
        print(f"{'='*60}")
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
            print(f"字符数: {len(chunk.page_content)}")
            print(f"内容:\n{chunk.page_content}")
            print("-" * 40)
        
        print(f"{'='*60}\n")
        
        content_preview = chunks[0].page_content[:500] if chunks else ""
        if len(chunks[0].page_content) > 500 if chunks else False:
            content_preview += "..."
        
        return UploadResponse(
            success=True,
            message="文件上传成功",
            file_id=file_id,
            filename=filename,
            document_count=len(documents),
            chunk_count=len(chunks),
            content_preview=content_preview
        )
        
    except Exception as e:
        # 删除已上传的文件
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@router.post("/analyze")
async def analyze_file(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """根据文件ID分析文档内容（支持多文件，流式输出）"""
    from crud import chat as chat_crud
    
    # 验证对话存在且属于当前用户
    conversation = chat_crud.get_conversation_by_id(db, request.conversation_id)
    if not conversation or conversation.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    async def generate(db):
        full_response = ""  # 累积完整响应
        is_first_message = chat_crud.get_conversation_messages(db, request.conversation_id, limit=1) == []
        
        try:
            file_ids = request.file_ids or []
            query = request.content
            
            if not query:
                yield f"data: {json.dumps({'error': '请提供分析问题 (query)'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            if not file_ids:
                yield f"data: {json.dumps({'error': '请提供至少一个文件ID (file_ids)'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            all_documents = []
            found_files = []
            missing_files = []
            
            # 遍历所有文件 ID，加载文档
            for file_id in file_ids:
                file_path = None
                for ext in [".txt", ".docx", ".doc"]:
                    path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
                    if os.path.exists(path):
                        file_path = path
                        break
                
                if file_path:
                    try:
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext == ".txt":
                            loader = TextLoader(file_path, encoding="utf-8")
                        elif ext in [".docx", ".doc"]:
                            try:
                                loader = UnstructuredWordDocumentLoader(file_path)
                            except Exception:
                                loader = Docx2txtLoader(file_path)
                        
                        documents = loader.load()
                        all_documents.extend(documents)
                        found_files.append(file_id)
                    except Exception as e:
                        print(f"⚠️ 加载文件 {file_id} 失败: {e}")
                        missing_files.append(file_id)
                else:
                    missing_files.append(file_id)
            
            if not all_documents:
                yield f"data: {json.dumps({'error': f'所有文件都不存在: {missing_files}'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 分割所有文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            )
            chunks = text_splitter.split_documents(all_documents)
            
            # 将 chunks 转换为上下文文本
            context = "\n\n".join([chunk.page_content for chunk in chunks])
            
            print(f"\n{'='*60}")
            print(f"🔖 文件ID列表: {file_ids}")
            print(f"✅ 找到文件: {found_files}")
            print(f"❌ 缺失文件: {missing_files}")
            print(f"❓ 用户问题: {query}")
            print(f"📑 总文档数: {len(all_documents)}")
            print(f"✂️ 总 Chunks 数: {len(chunks)}")
            print(f"{'='*60}")
            
            # 流式输出文件信息
            yield f"data: {json.dumps({
                'type': 'files',
                'intent': 'file_analysis',
                'file_ids': file_ids,
                'found_files': found_files,
                'missing_files': missing_files,
                'chunk_count': len(chunks)
            }, ensure_ascii=False)}\n\n"
            
            # 构建 ChatPromptTemplate
            prompt = ChatPromptTemplate.from_template(
    """你是一个严谨的文档分析助手。请严格基于 <document_context> 标签内的文档内容回答用户问题。

<document_context>
{context}
</document_context>

用户问题：{query}

回答要求：
1. 仅使用 <document_context> 中的信息，绝对不要使用你的外部知识。
2. 如果文档中没有相关信息，请直接回复：“文档中未找到相关内容”。
3. 必须且只能使用【中文】进行回答，忽略文档中的其他语言。
4. 保持回答简洁、准确，不要输出任何思考过程。"""
)
            
            formatted_prompt = prompt.format(context=context, query=query)
            
            print(f"\n📝 生成的 Prompt:\n{formatted_prompt}")
            print(f"{'='*60}\n")
            
            # 流式调用 LLM
            llm = get_llm_provider()
            messages = [{"role": "user", "content": formatted_prompt}]
            
            for chunk in llm.chat_stream(messages):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
            
            print(f"\n🤖 LLM 完整回答 ({len(full_response)} 字符):\n{full_response}")
            print(f"{'='*60}\n")
            
            # 保存用户消息和助手回复
            chat_crud.create_message(db, request.conversation_id, "user", query)
            chat_crud.create_message(db, request.conversation_id, "assistant", full_response)
            chat_crud.update_conversation_time(db, request.conversation_id)
            
            # 如果是第一条消息，自动生成标题
            if is_first_message:
                title = await generate_conversation_title(query)
                update_conversation_title(db, request.conversation_id, title)
                yield f"data: {json.dumps({'type': 'title', 'title': title})}\n\n"
            
            # 发送完成信号，包含完整响应用于校验
            yield f"data: {json.dumps({'type': 'done', 'full_content': full_response}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"❌ 流式输出异常: {e}")
            yield f"data: {json.dumps({'error': f'文件分析失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """删除已上传的文件"""
    deleted = False
    for ext in [".txt", ".docx", ".doc"]:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        if os.path.exists(file_path):
            os.unlink(file_path)
            deleted = True
            break
    
    if not deleted:
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_id}")
    
    return {"success": True, "message": "文件删除成功"}
