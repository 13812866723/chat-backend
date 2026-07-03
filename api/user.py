from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas.user import UserCreate, UserResponse, UserLogin, TokenResponse
from crud import user as user_crud
from config.database import SessionLocal, get_db
from config.security import verify_password, create_access_token, get_current_user, TokenData

router = APIRouter(prefix="/user", tags=["用户"])


@router.post("/register", response_model=TokenResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = user_crud.get_user_by_username(db, user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="用户名已存在")
    new_user = user_crud.create_user(db, user.username, user.password)
    access_token = create_access_token({"sub": str(new_user.id), "username": new_user.username})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(id=new_user.id, username=new_user.username)
    )


@router.post("/login", response_model=TokenResponse)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = user_crud.get_user_by_username(db, user.username)
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    access_token = create_access_token({"sub": str(db_user.id), "username": db_user.username})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(id=db_user.id, username=db_user.username)
    )
