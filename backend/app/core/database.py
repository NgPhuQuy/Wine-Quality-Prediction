import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Nạp các biến từ file .env vào hệ thống
load_dotenv()

# 2. Lấy URL từ biến môi trường (tên biến phải khớp với trong file .env)
# Nếu không tìm thấy, nó sẽ mặc định dùng chuỗi rỗng để tránh crash ngay lập tức
URL = os.getenv("DB_URL")

engine = create_engine(URL)

# Các phần dưới giữ nguyên
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally: 
        db.close()