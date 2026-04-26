from sqlalchemy import Column, Integer, String
from core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    
    # Sửa khúc này: Thêm độ dài và index để tìm kiếm nhanh hơn
    email = Column(String(100), unique=True, index=True, nullable=False)
    
    # Thêm cột Phone để khớp với giao diện Wine AI của nhóm
    phone = Column(String(20), nullable=True) 
    

    # thì độ dài vẫn đủ chứa chuỗi mã hóa.
    password = Column(String(255), nullable=False)