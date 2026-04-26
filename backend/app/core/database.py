import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

URL = os.getenv("DB_URL")

# FIX: Thêm kiểm tra URL trước khi tạo engine, tránh lỗi khó debug
if not URL:
    raise ValueError(
        "Biến môi trường DB_URL chưa được cấu hình. "
        "Hãy copy file .env.example thành .env và điền thông tin database."
    )

engine = create_engine(URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
