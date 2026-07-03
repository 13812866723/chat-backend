"""数据库配置"""
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# 加载环境变量
load_dotenv()
# 从环境变量中获取数据库URL
DATABASE_URL = os.getenv("DATABASE_URL")

# 创建数据库引擎
engine = create_engine(DATABASE_URL)
# 创建数据库会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# 创建SQLAlchemy模型基类
Base = declarative_base()


def init_db():
    """初始化数据库函数"""
    with engine.connect() as conn:
        # 创建vector扩展（用于向量搜索）
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        # 提交事务
        conn.commit()
    # 创建所有定义的表
    Base.metadata.create_all(bind=engine)


def check_db_connection():
    """检查数据库连接函数"""
    try:
        # 尝试连接数据库并执行简单查询
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db():
    """FastAPI 依赖注入函数 - 获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
