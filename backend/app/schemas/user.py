from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# 1. Base Schema: Chứa những gì cả lúc Tạo và lúc Trả về đều có
class UserBase(BaseModel):
    username: str
    email: EmailStr  # Chuyển thành bắt buộc vì trên giao diện có ô này
    phone: Optional[str] = None # Thêm trường này để khớp với Frontend

# 2. Schema dùng khi Đăng ký (Cần mật khẩu)
class UserCreate(UserBase):
    password: str

# 3. Schema dùng khi trả dữ liệu về (Không trả mật khẩu để bảo mật)
class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True # Giúp SQLAlchemy object chuyển sang JSON dễ dàng

class UserLogin(BaseModel):
    username: str
    password: str