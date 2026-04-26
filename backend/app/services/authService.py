from sqlalchemy.orm import Session
from ..models.user import User
from ..schemas.user import UserCreate
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_user_by_username(db: Session, username: str):
    """Tìm user theo username để kiểm tra trùng lặp hoặc đăng nhập"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str):
    """Tìm user theo email để đảm bảo email là duy nhất"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user: UserCreate):
    """Tạo và lưu user mới vào database"""
    hashed_pwd = pwd_context.hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        phone=user.phone,
        password=hashed_pwd
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate(db: Session, username: str, password: str):
    """Kiểm tra username và so khớp mật khẩu khi đăng nhập"""
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not pwd_context.verify(password, user.password):
        return None
    return user
