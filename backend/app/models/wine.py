from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class WinePrediction(Base):
    __tablename__ = "wine_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # Khóa ngoại nối với bảng User
    
    # Các thông số đầu vào (Features)
    fixed_acidity = Column(Float)
    volatile_acidity = Column(Float)
    citric_acid = Column(Float)
    residual_sugar = Column(Float)
    chlorides = Column(Float)
    free_sulfur_dioxide = Column(Float)
    total_sulfur_dioxide = Column(Float)
    density = Column(Float)
    ph = Column(Float)
    sulphates = Column(Float)
    alcohol = Column(Float)
    
    # Kết quả từ Model AI
    quality_score = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())