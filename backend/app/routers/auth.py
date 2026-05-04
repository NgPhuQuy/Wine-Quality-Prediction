from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.user import UserCreate, UserResponse, UserLogin
from ..services import authService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Kiểm tra username đã tồn tại chưa
    if authService.get_user_by_username(db, user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username đã tồn tại"
        )
    # Kiểm tra email đã tồn tại chưa
    if authService.get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã được sử dụng"
        )
    return authService.create_user(db, user)


@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    auth_user = authService.authenticate(db, user.username, user.password)

    if not auth_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tài khoản hoặc mật khẩu, kiểm tra lại nhé"
        )


    return {
        "message": "Đăng nhập thành công",
        "user_id": auth_user.id,
        "username": auth_user.username,
    }


@router.post("/logout")
def logout():
    return {
        "status": "success",
        "message": "Đăng xuất thành công."
    }
