# chat-backend

基于 FastAPI 的智能对话后端，集成大模型对话、RAG 本地知识库问答、联网搜索、意图识别与多步任务编排，支持短期记忆持久化。

## 目录结构

```
chat-backend/
├── api/                        # API 路由层
│   ├── user.py                 # 用户注册/登录
│   ├── chat.py                 # 基础聊天（含流式）
│   ├── intent_chat.py          # 意图识别聊天
│   ├── agent.py                # Agent 统一聊天 + 任务编排流式接口
│   ├── rag.py                  # RAG 向量库管理
│   └── upload.py               # 文件上传与分析
├── config/                     # 配置层
│   ├── database.py             # SQLAlchemy 引擎/会话/建表
│   ├── security.py             # 密码哈希(bcrypt) + JWT 认证
│   ├── llm.py                  # LLM 配置
│   └── rag.py                  # RAG / Embedding 配置
├── providers/                  # 大模型提供商
│   ├── base.py                 # BaseLLMProvider 抽象基类
│   ├── factory.py              # get_llm_provider() 工厂
│   ├── siliconflow.py          # 硅基流动
│   └── zhipu.py                # 智谱 AI
├── services/                   # 业务服务层
│   ├── agent.py                # 统一 Agent + PostgreSQL 短期记忆
│   ├── task_graph.py           # LangGraph 任务编排图（意图引导→规划→执行）
│   ├── intent_recognition.py   # 意图识别服务
│   └── utils.py                # 工具函数（标题生成等）
├── rag/                        # RAG 服务
│   ├── service.py              # RAGService + rag_chat_tool
│   ├── embedding.py            # SiliconFlow 嵌入模型
│   └── document_loader.py      # 文档加载器
├── tools/                      # LangChain 工具
│   └── tavily.py               # Tavily 联网搜索工具
├── crud/                       # 数据库操作层
│   ├── user.py
│   └── chat.py
├── schemas/                    # Pydantic 数据模型
│   ├── user.py
│   ├── chat.py
│   └── upload.py
├── documents/                  # 本地知识库文档目录
├── chroma_data/                # Chroma 向量库持久化目录（自动生成）
├── uploads/                    # 上传文件暂存目录（启动时自动清理）
├── models.py                   # SQLAlchemy ORM 模型
├── main.py                     # FastAPI 应用入口
└── requirements.txt
```

## 核心能力

### 1. 统一 Agent（RAG + 联网搜索 + 直接回答）
[services/agent.py](services/agent.py) 基于 LangChain `create_agent` 构建统一 Agent，绑定两个工具：
- `rag_chat_tool` — 查询本地知识库（民法典合同编）
- `tavily_search_tool` — 联网获取实时信息

Agent 根据 system prompt 自主判断：意图判断 → 流程规划 → 执行，模糊输入会先追问。

### 2. 任务编排图（结构化多步规划）
[services/task_graph.py](services/task_graph.py) 基于 LangGraph StateGraph 实现显式编排：

```
START → entry_route ──(上轮在等追问)──► plan
          │
          │ (新输入)
          ▼
   classify_intent
          │
   route_after_classify ──(置信度低)──► clarify ──► END
          │ (意图清晰)
          ▼
        plan ──► execute_step ──► route_after_execute
                                      ├─(还有步骤)─► execute_step (循环)
                                      └─(全部完成)─► finalize ──► END
```

- **意图引导**：置信度 < 0.6 时生成追问，下一轮自动合并原始问题与补充回答
- **多步执行**：规划器拆解 1-3 步（`rag:` / `search:` / `answer:` 前缀），逐步执行
- **流式汇总**：finalize 节点通过 LangGraph custom stream writer 逐 token 推送

### 3. 短期记忆（PostgreSQL Checkpointer）
[services/agent.py](services/agent.py) 使用 `AsyncPostgresSaver` 将对话状态按 `thread_id`（即 `conversation_id`）持久化到 PostgreSQL：
- 同一对话的多轮消息自动加载历史上下文
- 服务重启后记忆依旧保留
- 初始化在 `main.py` 启动时完成（`init_checkpointer()`）

> 注意：checkpointer 状态与 `chat_messages` 业务表是两套独立存储——前者供 Agent 推理，后者供前端展示。

### 4. RAG 本地知识库
[rag/service.py](rag/service.py) 使用 Chroma 向量库：
- 启动时自动从 `./documents/` 加载文档（.txt/.md/.json）
- 相似度阈值过滤（默认 0.4，可配置）
- 检索不到合格文档时直接返回提示，不调用 LLM 兜底

## 数据库

### 表结构

**users**
| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer | 主键 |
| username | String(50) | 用户名，唯一 |
| hashed_password | String(255) | bcrypt 哈希 |
| created_at | DateTime | 创建时间 |

**conversations**
| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer | 主键 |
| user_id | Integer | 外键 → users.id |
| title | String(100) | 对话标题（首条消息自动生成） |
| created_at | DateTime | 创建时间 |
| updated_at | DateTime | 更新时间 |

**chat_messages**
| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer | 主键 |
| conversation_id | Integer | 外键 → conversations.id |
| role | String(20) | 'user' 或 'assistant' |
| content | Text | 消息内容 |
| created_at | DateTime | 创建时间 |

> LangGraph checkpointer 会额外创建 `checkpoints` / `checkpoint_writes` / `checkpoint_blobs` 表，由 `AsyncPostgresSaver.setup()` 自动管理。

## API 接口

所有接口（除 `/` 和 `/health`）需在 Header 携带 `Authorization: Bearer <access_token>`。

### 用户
| 路由 | 方法 | 说明 |
|------|------|------|
| `/user/register` | POST | 注册，返回 JWT |
| `/user/login` | POST | 登录，返回 JWT |

### 基础聊天
| 路由 | 方法 | 说明 |
|------|------|------|
| `/chat/conversation/create` | POST | 创建对话 |
| `/chat/conversation/list` | GET | 对话列表 |
| `/chat/conversation/{id}/history` | GET | 历史消息 |
| `/chat/message/save` | POST | 保存消息 |
| `/chat/chat` | POST | 同步聊天 |
| `/chat/stream` | POST | 流式聊天（SSE） |

### Agent 聊天
| 路由 | 方法 | 说明 |
|------|------|------|
| `/chat/agent` | POST | Agent 统一聊天（非流式，含短期记忆） |
| `/chat/agent/stream` | POST | Agent 流式聊天（SSE，含工具调用状态推送） |
| `/chat/task-stream` | POST | 任务编排流式（SSE，意图→规划→执行→汇总） |

### 意图识别聊天
| 路由 | 方法 | 说明 |
|------|------|------|
| `/chat/intent-chat` | POST | 意图识别聊天 |
| `/chat/intent-chat/stream` | POST | 流式意图聊天 |
| `/chat/intent-test` | POST | 仅测试意图识别 |

### RAG 管理
| 路由 | 方法 | 说明 |
|------|------|------|
| `/rag/reload` | POST | 重新加载向量库 |
| `/rag/status` | GET | 向量库状态 |

### 文件上传
| 路由 | 方法 | 说明 |
|------|------|------|
| `/upload/file` | POST | 上传文件（.txt/.docx） |
| `/upload/analyze` | POST | 基于上传文件流式问答 |
| `/upload/{file_id}` | DELETE | 删除文件 |

### SSE 事件类型（Agent / Task-stream）

`/chat/agent/stream` 推送：
- `{"type":"content","content":"..."}` — 文本增量
- `{"type":"tool","tool_name":"...","content":"..."}` — 工具调用状态
- `{"type":"sources","sources":"..."}` — 工具返回来源
- `{"type":"title","title":"..."}` — 首条消息自动生成标题

`/chat/task-stream` 推送：
- `{"type":"intent","intent":"...","confidence":0.9}` — 意图识别结果
- `{"type":"clarify","content":"..."}` — 追问（本轮结束）
- `{"type":"plan","steps":["rag:...","answer:..."]}` — 规划步骤
- `{"type":"status","step":"...","content":"..."}` — 步骤执行完毕
- `{"type":"content","content":"..."}` — 最终回答（多步时逐 token 流式）
- `{"type":"title","title":"..."}` — 首条消息自动生成标题

## 环境变量

复制 `.env` 并按实际填写：

```env
# 数据库
DATABASE_URL=postgresql://postgres:123456@localhost:5432/chat

# JWT 密钥（生产环境务必修改）
SECRET_KEY=your-secret-key-change-in-production

# 大模型配置
LLM_PROVIDER=siliconflow              # siliconflow / zhipu
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen3.5-4B
LLM_TIMEOUT=120

# RAG 配置
EMBEDDING_API_KEY=your-embedding-api-key
RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RAG_SIMILARITY_THRESHOLD=0.4
RAG_TOP_K=3
RAG_CHUNK_SIZE=400
RAG_CHUNK_OVERLAP=100
RAG_DOCUMENTS_DIR=./documents
CHROMA_PERSIST_DIR=./chroma_data

# Tavily 联网搜索
TAVILY_API_KEY=your-tavily-api-key
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备 PostgreSQL 数据库
createdb chat

# 3. 配置环境变量
cp .env.example .env  # 按实际填写

# 4. 放置知识库文档到 ./documents/

# 5. 启动
python main.py
# 或
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## JWT 认证

- 登录/注册成功返回 `access_token`
- 后续请求 Header：`Authorization: Bearer <access_token>`
- Token 有效期：**1 小时**（`ACCESS_TOKEN_EXPIRE_MINUTES`，可配置）
- Token payload：`sub`(用户ID)、`username`、`exp`

## 大模型 Provider 扩展

1. 在 `providers/` 下新建类，继承 `BaseLLMProvider`，实现 `chat` / `chat_stream`
2. 在 [providers/factory.py](providers/factory.py) 添加 provider 分支
3. `.env` 设置 `LLM_PROVIDER`

```python
from providers.factory import get_llm_provider

llm = get_llm_provider()
response = llm.chat([
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
])
```

## 已知问题

- `api/intent_chat.py` 的 `intent_based_chat` 用裸 query 参数（`message: str, conversation_id: int`），与其他端点的 JSON body 风格不一致
- `providers/base.py` 的 `bind_tools` 未声明为抽象方法，仅 SiliconFlowProvider 实现
