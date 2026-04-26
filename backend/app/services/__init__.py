import joblib
import os
from pathlib import Path

# --- 1. XỬ LÝ ĐƯỜNG DẪN ĐỘNG (DYNAMIC PATH) ---
# Lấy vị trí file hiện tại và tìm về thư mục gốc của project
current_path = Path(__file__).resolve()
# . là services -> .. là app -> ... là backend -> .... là gốc Wine-Quality-Prediction
root_dir = current_path.parent.parent.parent.parent

# --- 2. ĐỊNH NGHĨA ĐƯỜNG DẪN CHUẨN (Sửa hết typo ở đây) ---
MODEL_FILE = root_dir / "models" / "final_model.joblib"
# Quý kiểm tra file thật trong folder metadata xem tên là gì nhé (thường là .joblib)
METADATA_FILE = root_dir / "metadata" / "final_metadata.joblib" 

# --- 3. NẠP MODEL VÀ METADATA ---
model = None
metadata = None

if MODEL_FILE.exists():
    model = joblib.load(str(MODEL_FILE))
    print(f"✅ Đã nạp Model thành công từ: {MODEL_FILE}")
else:
    print(f"⚠️ Cảnh báo: Không tìm thấy file model tại: {MODEL_FILE}")

if METADATA_FILE.exists():
    metadata = joblib.load(str(METADATA_FILE))
    print(f"✅ Đã nạp Metadata thành công từ: {METADATA_FILE}")
else:
    # Quý kiểm tra lại tên file thật ngoài folder xem có đúng là 'wine_metadata.joblib' không
    print(f"⚠️ Cảnh báo: Không tìm thấy metadata tại: {METADATA_FILE}")

# --- 4. DANH SÁCH CỘT (Dùng cho Predict Service) ---
COLUMNS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'ph', 'sulphates', 'alcohol', 'type', 
    'total_acidity', 'sugar_alcohol_ratio', 'density_alcohol_interaction'
]