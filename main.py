from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from config.database import engine, Base, check_db_connection
from api.user import router as user_router
from api.chat import router as chat_router
from api.intent_chat import router as intent_chat_router
from api.rag import router as rag_router
from api.upload import router as upload_router
from api.agent import router as agent_router
from services.agent import init_checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：建表 + 初始化 Agent 短期记忆 Checkpointer
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表自动创建成功！")
    await init_checkpointer()
    print("✅ Agent Checkpointer（PostgreSQL 短期记忆）初始化成功！")
    yield
    # 关闭：目前无额外清理


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(chat_router)
app.include_router(intent_chat_router)
app.include_router(rag_router)
app.include_router(upload_router)
app.include_router(agent_router)


@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/health")
def health_check():
    db_status = "connected" if check_db_connection() else "disconnected"
    return {
        "status": "healthy",
        "database": db_status
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
