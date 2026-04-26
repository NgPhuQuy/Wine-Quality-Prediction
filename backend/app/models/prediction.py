# app/models/prediction.py
from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # Liên kết với bảng User
    
    # Thông số rượu (đầu vào)
    alcohol = Column(Float)
    volatile_acidity = Column(Float)
    ph = Column(Float)
    # ... thêm các field khác ...
    
    # Kết quả AI (đầu ra)
    quality_score = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())