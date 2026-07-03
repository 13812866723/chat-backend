from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from config.rag import RAGConfig
from rag.embedding import SiliconFlowEmbeddings
from providers.factory import get_llm_provider
import os
import uuid
import json


class RAGService:
    """RAG 服务类 - 使用 Chroma 向量数据库持久化存储"""
    
    def __init__(self):
        # 初始化嵌入模型
        self.embeddings = SiliconFlowEmbeddings(RAGConfig.EMBEDDING_MODEL)
        print(f"✅ 已初始化嵌入模型 {RAGConfig.EMBEDDING_MODEL}")
        # 初始化 Chroma 向量库（持久化模式）
        persist_dir = RAGConfig.CHROMA_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name="rag_documents"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", ""], 
        )
        self.llm = get_llm_provider()
        
        # 加载状态跟踪
        self.load_error = None
        
        # 启动时加载本地文档（如果需要）
        if self.vectorstore._collection.count() == 0:
            print("📚 Chroma 向量库为空，正在从本地文件加载文档...")
            self.load_documents_from_files()
        else:
            print(f"✅ 已从 Chroma 加载 {self.vectorstore._collection.count()} 个文档片段")
    
    def is_available(self) -> bool:
        """检查 RAG 是否可用（有文档且无加载错误）"""
        return self.vectorstore._collection.count() > 0 and self.load_error is None
    
    def load_documents_from_files(self):
        """从本地目录加载文档到 Chroma"""
        docs_dir = RAGConfig.DOCUMENTS_DIR
        if not os.path.exists(docs_dir):
            print(f"⚠️ 文档目录 {docs_dir} 不存在，跳过加载")
            return
        
        supported_extensions = ['.txt', '.md', '.json']
        doc_id = 0
        self.load_error = None  # 重置错误状态
        
        for filename in os.listdir(docs_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                continue
            
            file_path = os.path.join(docs_dir, filename)
            try:
                # 使用 TextLoader 加载文件
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                # 使用 split_documents 分割文档
                chunks = self.text_splitter.split_documents(docs)               
                
                # print(f"原始文档数量: {len(docs)}")
                # print(f"分割后块数量: {len(chunks)}\n")

                # for i, chunk in enumerate(chunks):
                #     print(f"--- Chunk {i+1} ---")
                #     print(f"内容: {chunk.page_content}")
                #     print(f"元数据: {chunk.metadata}\n")

                # 只打印前几个
                for i, chunk in enumerate(chunks[:3]):  # 只显示前5个块
                    print(f"--- Chunk {i+1} ---")
                    print(f"内容: {chunk.page_content}")  # 只显示前200个字符
                    print(f"元数据: {chunk.metadata}\n")

                # 为每个 chunk 生成 ID
                ids = [str(uuid.uuid4()) for _ in chunks]
                
                # 使用 add_documents 添加到向量库
                self.vectorstore.add_documents(
                    documents=chunks,
                    ids=ids
                )
                
                print(f"✅ 加载文档: {filename}，分割为 {len(chunks)} 个片段")

                doc_id += len(chunks)
            
            except Exception as e:
                self.load_error = str(e)  # 记录加载错误
                print(f"❌ 加载文档 {filename} 失败: {e}")
                # 1. 打印基础的异常信息
                print(f"捕获到异常类型: {type(e).__name__}")
                print(f"基础异常信息: {e}")

                # 2. 尝试获取 HTTP 响应体（这是最关键的一步，API的具体报错通常在这里）
                if hasattr(e, 'response') and e.response is not None:
                    print(f"HTTP 状态码: {e.response.status_code}")
                    print(f"API 详细错误信息: {e.response.text}")
                    # 如果返回的是 JSON 格式，也可以尝试解析
                    try:
                        print(f"API JSON 响应: {e.response.json()}")
                    except ValueError:
                        pass
                else:
                    print("该异常没有包含 HTTP 响应对象")
        
        print(f"📚 向量库共 {self.vectorstore._collection.count()} 个文档片段")
    
    def retrieve_relevant_docs(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """检索与查询最相关的文档"""
        k = top_k or RAGConfig.TOP_K
        
        print(f"\n[DEBUG] === 开始检索 ===")
        print(f"[DEBUG] 查询语句: '{query}'")
        print(f"[DEBUG] 请求检索数量 top_k: {k}")
   
        # 从 Chroma 查询相似文档（直接返回相似度分数）
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k
        )

        print(f"[DEBUG] 向量数据库原始返回文档数: {len(results)}")
        
        if not results:
            print("[DEBUG] ⚠️ 未检索到任何相关文档，返回空列表。")
            return []
        
        docs = []
        for i, (doc, similarity) in enumerate(results):
            # similarity_search_with_relevance_scores 直接返回相似度分数
            is_kept = similarity >= RAGConfig.SIMILARITY_THRESHOLD
            status = "✅ 保留" if is_kept else "❌ 过滤"
            print(f"[DEBUG] 文档 {i+1} | 相似度: {similarity:.4f} | 阈值: {RAGConfig.SIMILARITY_THRESHOLD} | {status}")
            print(f"        内容预览: '{doc.page_content[:50]}...'") # 只打印前50个字符避免刷屏
            
            if is_kept:
                docs.append({
                    "id": doc.metadata.get("chunk_id", ""),
                    "content": doc.page_content,
                    "similarity": similarity,
                    "source": doc.metadata.get("source", "unknown")
                })
        
        # 按相似度降序排序
        docs.sort(key=lambda x: x["similarity"], reverse=True)
        print(f"[DEBUG] 过滤后保留的文档数: {len(docs)}")
        if docs:
            print("[DEBUG] 最终文档相似度排序:")
            for i, d in enumerate(docs):
                print(f"        Top {i+1}: 相似度 {d['similarity']:.4f} | 来源: {d['source']} | ID: {d['id']}")
        else:
            print("[DEBUG] ⚠️ 所有文档均低于相似度阈值，无有效文档返回。")
            
        print(f"[DEBUG] === 检索结束 ===\n")
        return docs
    
    def build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """构建带有上下文的提示词"""
        context = "\n\n".join([doc["content"] for doc in context_docs])
        
        prompt = f"""
基于以下上下文信息回答用户问题：

{context}

---

用户问题：{query}

请根据提供的上下文信息回答问题。如果上下文中没有相关信息，请说明"根据现有信息无法回答该问题"。
"""
        return prompt.strip()
    
    def chat_with_context(self, query: str) -> str:
        """使用 RAG 进行聊天"""
        # 检索相关文档
        relevant_docs = self.retrieve_relevant_docs(query)
        
        if not relevant_docs:
            # 未检索到合格文档，直接告知用户
            return "\n\n抱歉，未在知识库中检索到与该问题相关的内容。"
        
        # 构建带上下文的提示词
        prompt = self.build_prompt(query, relevant_docs)
        messages = [{"role": "user", "content": prompt}]
        
        return self.llm.chat(messages)
    
    def chat_stream_with_context(self, query: str):
        """使用 RAG 进行流式聊天"""
        # 检索相关文档
        relevant_docs = self.retrieve_relevant_docs(query)
        print(f"检索到 {len(relevant_docs)} 个相关文档")
        for doc in relevant_docs:
            print(f"文档 ID: {doc['id']}, 相似度: {doc['similarity']:.4f}, 来源: {doc['source']}")

        # 未检索到合格文档，返回提示（与 chat_with_context 行为一致）
        if not relevant_docs:
            yield json.dumps({"content": "\n\n抱歉，未在知识库中检索到与该问题相关的内容。"}, ensure_ascii=False)
            yield "[DONE]"
            return

        # 流式输出引用来源
        sources_data = {
            "type": "sources",
            "sources": [
                {
                    "id": doc["id"],
                    "source": doc["source"],
                    "similarity": round(doc["similarity"], 4),
                    "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                }
                for doc in relevant_docs
            ]
        }
        yield json.dumps(sources_data, ensure_ascii=False)

        # 构建带上下文的提示词
        prompt = self.build_prompt(query, relevant_docs)
        messages = [{"role": "user", "content": prompt}]

        # 流式输出 LLM 响应
        for chunk in self.llm.chat_stream(messages):
            yield json.dumps({"content": chunk}, ensure_ascii=False)
        yield "[DONE]"
    
    def get_documents_count(self) -> int:
        """获取向量库中的文档片段数量"""
        return self.vectorstore._collection.count()
    
    def reload_documents(self):
        """重新加载文档"""
        # 删除现有向量库
        self.vectorstore.delete_collection()
        # 重新创建向量库
        self.vectorstore = Chroma(
            persist_directory=RAGConfig.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="rag_documents"
        )
        # 重新加载
        self.load_documents_from_files()


# 全局 RAGService 实例（供 Tool 使用）
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """获取或创建全局 RAGService 实例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@tool
def rag_chat_tool(query: str) -> str:
    """
    RAG 聊天工具 - 基于本地知识库检索增强生成回答。

    本地知识库包含以下文档：
    - 中华人民共和国民法典合同编

    当需要回答关于合同编法条、民事法律问题、文档内容等相关问题时使用此工具。
    该工具会从向量数据库中检索相关文档，并结合检索结果生成回答。

    参数:
        query: 用户的查询问题

    返回:
        结合检索到的文档上下文生成的回答文本
    """
    rag_service = get_rag_service()

    if not rag_service.is_available():
        return "抱歉，知识库当前不可用或尚未加载任何文档。"

    return rag_service.chat_with_context(query)
