from sqlalchemy.orm import Session
from app.models.wine import WinePrediction
from app.schemas.wine import WineCreate
from . import model, COLUMNS # Đảm bảo file __init__.py trong services đã load model
import pandas as pd

def predict_and_store_wine(db: Session, data: WineCreate, user_id: int):
    try:
        # 1. Chuyển Pydantic Schema sang Dictionary
        input_data = data.model_dump()

        # 2. Trích xuất giá trị (Dùng đúng tên biến trong Schema ph, wine_type)
        f_acid = float(input_data.get("fixed_acidity", 0))
        v_acid = float(input_data.get("volatile_acidity", 0))
        res_sugar = float(input_data.get("residual_sugar", 0))
        alc = float(input_data.get("alcohol", 0))
        dens = float(input_data.get("density", 0))
        
        # 3. Feature Engineering (Giữ nguyên logic của nhóm)
        total_acidity = f_acid + v_acid
        sugar_alcohol_ratio = res_sugar / (alc + 1e-5)
        density_alcohol_interaction = dens * alc

        # 4. Tạo Dictionary đầu vào cho DataFrame (Phải khớp 100% với tên cột lúc Train)
        input_dict = {
            'fixed acidity': f_acid,
            'volatile acidity': v_acid,
            'citric acid': float(input_data.get("citric_acid", 0)),
            'residual sugar': res_sugar,
            'chlorides': float(input_data.get("chlorides", 0)),
            'free sulfur dioxide': float(input_data.get("free_sulfur_dioxide", 0)),
            'total sulfur dioxide': float(input_data.get("total_sulfur_dioxide", 0)),
            'density': dens,
            'pH': float(input_data.get("ph", 0)), # Chú ý: ph (schema) -> pH (model)
            'sulphates': float(input_data.get("sulphates", 0)),
            'alcohol': alc,
            'type': int(input_data.get("wine_type", 0)), # wine_type (schema) -> type (model)
            'total_acidity': total_acidity,
            'sugar_alcohol_ratio': sugar_alcohol_ratio,
            'density_alcohol_interaction': density_alcohol_interaction
        }

        # 5. Ép kiểu DataFrame và kiểm tra thứ tự cột theo mảng COLUMNS
        df = pd.DataFrame([input_dict])[COLUMNS]

        # 6. Thực hiện Dự đoán
        prediction = int(model.predict(df)[0])
        # Kiểm tra nếu model có hỗ trợ xác suất (predict_proba)
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.0

        # 7. LƯU VÀO DATABASE
        db_prediction = WinePrediction(
            user_id=user_id,
            fixed_acidity=f_acid,
            volatile_acidity=v_acid,
            citric_acid=float(input_data.get("citric_acid")),
            residual_sugar=res_sugar,
            chlorides=float(input_data.get("chlorides")),
            free_sulfur_dioxide=float(input_data.get("free_sulfur_dioxide")),
            total_sulfur_dioxide=float(input_data.get("total_sulfur_dioxide")),
            density=dens,
            ph=float(input_data.get("ph")),
            sulphates=float(input_data.get("sulphates")),
            alcohol=alc,
            quality_score=prediction 
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return {
            "status": "success",
            "id": db_prediction.id, 
            "quality_score": prediction,
            "raw_score": round(proba, 4),
            "quality_label": "Good Wine (>=7)" if prediction == 1 else "Normal Wine (<7)",
            "created_at": db_prediction.created_at
        }

    except Exception as e:
        db.rollback() # Quan trọng: Rollback nếu lỗi để tránh treo DB
        return {
            "status": "error",
            "message": f"Lỗi xử lý dự đoán: {str(e)}"
        }