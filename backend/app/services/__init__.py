import joblib
from pathlib import Path

current_path = Path(__file__).resolve()
root_dir = current_path.parent.parent.parent.parent

MODEL_FILE = root_dir / "models" / "final_model.joblib"
METADATA_FILE = root_dir / "metadata" / "final_metadata.joblib"

model = None
metadata = None

if MODEL_FILE.exists():
    model = joblib.load(str(MODEL_FILE))
    print(f"Nạp Model thành công từ: {MODEL_FILE}")
else:
    print(f"Cảnh báo: Không tìm thấy file model tại: {MODEL_FILE}")

if METADATA_FILE.exists():
    metadata = joblib.load(str(METADATA_FILE))
    print(f"Nạp Metadata thành công từ: {METADATA_FILE}")
else:
    print(f"Cảnh báo: Không tìm thấy metadata tại: {METADATA_FILE}")

COLUMNS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'type', 
    'total_acidity', 'sugar_alcohol_ratio', 'density_alcohol_interaction'
] 