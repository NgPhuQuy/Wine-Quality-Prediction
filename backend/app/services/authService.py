from sqlalchemy.orm import Session
from ..models.user import User
from ..schemas.user import UserCreate
from passlib.context import CryptContext

# Khai báo "máy băm" mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- 1. HÀM TÌM KIẾM (Cái này Router đang đòi đây Quý!) ---

def get_user_by_username(db: Session, username: str):
    """Tìm user theo username để kiểm tra trùng lặp hoặc đăng nhập"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    """Tìm user theo email để đảm bảo email là duy nhất"""
    return db.query(User).filter(User.email == email).first()

# --- 2. HÀM TẠO USER ---

def create_user(db: Session, user: UserCreate):
    """Băm mật khẩu và lưu user mới vào database"""
    # Bước A: Băm mật khẩu (An toàn tuyệt đối)
    hashed_pwd = pwd_context.hash(user.password) 
    
    # Bước B: Khai báo đối tượng User từ model
    db_user = User(
        username=user.username,
        email=user.email,
        phone=user.phone,
        password=hashed_pwd 
    )
    
    # Bước C: Lưu xuống MySQL
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- 3. HÀM XÁC THỰC ---

def authenticate(db: Session, username: str, password: str):
    """Kiểm tra username và so khớp mật khẩu khi đăng nhập"""
    user = get_user_by_username(db, username)
    if not user:
        return None
    
    # So khớp mật khẩu nhập vào với mật khẩu đã băm trong database
    if not pwd_context.verify(password, user.password):
        return None
    return user