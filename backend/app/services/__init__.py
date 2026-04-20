import joblib
from ..core.config import MODEL_PATH, METADATA_PATH

# 1. Load model
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

# 2. ĐỊNH NGHĨA TÊN CỘT
COLUMNS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'type', 
    'total_acidity', 'sugar_alcohol_ratio', 'density_alcohol_interaction'
]
