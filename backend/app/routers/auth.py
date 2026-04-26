from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from core.database import get_db
from schemas.user import UserCreate, UserResponse, UserLogin # Thêm UserLogin vào đây
from services import authService

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # 1. Kiểm tra xem tên đăng nhập đã bị ai lấy chưa
    db_user = authService.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username này đã có người sử dụng rồi Quý ơi!"
        )
    
    # 2. Gọi service để tạo user mới (nhớ kiểm tra service đã nhận đủ 4 trường chưa nhé)
    return authService.create_user(db, user)

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)): # Dùng UserLogin ở đây
    # 3. Chỉ nhận username và password để xác thực
    auth_user = authService.authenticate(db, user.username, user.password)
    
    if not auth_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Sai tài khoản hoặc mật khẩu, kiểm tra lại nhé!"
        )
    
    # 4. Trả về thông báo thành công và thông tin cơ bản
    return {
        "message": "Đăng nhập thành công", 
        "username": auth_user.username,
        
    }