import joblib
import pandas as pd

# 1. Load model
model = joblib.load("models/rf_model.joblib")


# 2. ĐỊNH NGHĨA TÊN CỘT
COLUMNS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'type', 
    'total_acidity', 'sugar_alcohol_ratio', 'density_alcohol_interaction'
]

def predict_wine(data):
    try:
        # Chuyển data sang dạng dict nếu nó là object (để dùng .get cho an toàn)
        if hasattr(data, '__dict__'):
            data = data.__dict__
        else:
            data = data

        # Lấy giá trị và xử lý lỗi tên biến có/không có gạch dưới
        f_acid = float(data.get("fixed_acidity") or data.get("fixed acidity") or 0)
        v_acid = float(data.get("volatile_acidity") or data.get("volatile acidity") or 0)
        res_sugar = float(data.get("residual_sugar") or data.get("residual sugar") or 0)
        alc = float(data.get("alcohol") or 0)
        dens = float(data.get("density") or 0)
        
        # Feature Engineering (giữ nguyên logic của bạn)
        total_acidity = f_acid + v_acid
        sugar_alcohol_ratio = res_sugar / (alc + 1e-5)
        density_alcohol_interaction = dens * alc

        # TẠO DICT VỚI KEY CÓ KHOẢNG TRẮNG (Để khớp với COLUMNS)
        input_dict = {
            'fixed acidity': f_acid,
            'volatile acidity': v_acid,
            'citric acid': float(data.get("citric_acid") or data.get("citric acid") or 0),
            'residual sugar': res_sugar,
            'chlorides': float(data.get("chlorides") or 0),
            'free sulfur dioxide': float(data.get("free_sulfur_dioxide") or data.get("free sulfur dioxide") or 0),
            'total sulfur dioxide': float(data.get("total_sulfur_dioxide") or data.get("total sulfur dioxide") or 0),
            'density': dens,
            'pH': float(data.get("pH") or 0),
            'sulphates': float(data.get("sulphates") or 0),
            'alcohol': alc,
            'type': 0 if str(data.get("type", "red")).lower() == "red" else 1,
            'total_acidity': total_acidity,
            'sugar_alcohol_ratio': sugar_alcohol_ratio,
            'density_alcohol_interaction': density_alcohol_interaction
        }

        # Ép kiểu DataFrame và kiểm tra thứ tự cột
        df = pd.DataFrame([input_dict])[COLUMNS]

        # Thực hiện predict
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        return {
            "status": "success",
            "is_good_wine": bool(prediction),
            "raw_score": round(float(proba), 4),
            "quality_label": "Good (>=7)" if prediction == 1 else "Normal (<7)"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Lỗi: {str(e)}"
        }

# --- CHẠY TEST ---
if __name__ == "__main__":
    test_data = {
    "fixed_acidity": 10000,
    "volatile_acidity": 10000,
    "citric_acid": 10000,
    "residual_sugar": 10000,
    "chlorides": 10000,
    "free_sulfur_dioxide": 10000,
    "total_sulfur_dioxide": 10000,
    "density": 1100110,
    "pH": 10000,
    "sulphates": 10000,
    "alcohol": 10000,
    "type": "white"
}
    
    print(predict_wine(test_data))