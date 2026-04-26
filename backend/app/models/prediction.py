# FIX: Đổi từ absolute import ("app.core.database") sang relative import ("..core.database")
# để tránh lỗi ModuleNotFoundError khi chạy từ thư mục backend
# File này hiện không được dùng trực tiếp (WinePrediction đã định nghĩa đủ trong wine.py)
# Giữ lại để tham khảo, nhưng import phải đúng

from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from sqlalchemy.sql import func
from ..core.database import Base  # FIX: absolute "app.core.database" -> relative "..core.database"

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    alcohol = Column(Float)
    volatile_acidity = Column(Float)
    ph = Column(Float)
    quality_score = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
