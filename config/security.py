"""安全配置"""
import os
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 密码加密上下文，使用 bcrypt 算法
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 从环境变量获取密钥，如果没有则使用默认值（生产环境需要修改）
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"  # JWT 使用的加密算法
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 1  # 1 小时


# 这是一个使用Pydantic库定义的TokenData模型类
# 用于表示令牌(Token)相关的数据结构
class TokenData(BaseModel):
    user_id: int    # 用户ID，整数类型
    username: str   # 用户名，字符串类型


# 导入HTTPBearer用于验证HTTP授权凭据
security = HTTPBearer()


# 密码验证和哈希相关函数
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict) -> str:
    # 1. 复制传入的数据字典
    to_encode = data.copy()
    
    # 2. 设置过期时间
    # 获取当前的 UTC 时间，并加上配置的过期分钟数
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # 3. 将过期时间添加到待编码的数据中
    to_encode.update({"exp": expire})
    
    # 4. 使用 PyJWT 库进行编码生成 Token
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)



def decode_access_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """解析请求头中的 Token，有效则返回用户信息，无效返回 401"""
    token = credentials.credentials
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub"))
        username = payload.get("username")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token 无效")
        return TokenData(user_id=user_id, username=username)
    except Exception:
        raise HTTPException(status_code=401, detail="Token 已过期或无效")
