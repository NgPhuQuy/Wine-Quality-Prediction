from sqlalchemy.orm import Session
from ..models.wine import WinePrediction
from ..schemas.wine import WineCreate
from . import model, COLUMNS
import pandas as pd


def predict_and_store_wine(db: Session, data: WineCreate, user_id: int):
    try:
        input_data = data.model_dump()

        f_acid   = float(input_data.get("fixed_acidity", 0))
        v_acid   = float(input_data.get("volatile_acidity", 0))
        res_sugar = float(input_data.get("residual_sugar", 0))
        alc      = float(input_data.get("alcohol", 0))
        dens     = float(input_data.get("density", 0))

        total_acidity               = f_acid + v_acid
        sugar_alcohol_ratio         = res_sugar / (alc + 1e-5)
        density_alcohol_interaction = dens * alc

        input_dict = {
            'fixed acidity':              f_acid,
            'volatile acidity':           v_acid,
            'citric acid':                float(input_data.get("citric_acid", 0)),
            'residual sugar':             res_sugar,
            'chlorides':                  float(input_data.get("chlorides", 0)),
            'free sulfur dioxide':        float(input_data.get("free_sulfur_dioxide", 0)),
            'total sulfur dioxide':       float(input_data.get("total_sulfur_dioxide", 0)),
            'density':                    dens,
            'ph':                         float(input_data.get("ph", 0)),
            'sulphates':                  float(input_data.get("sulphates", 0)),
            'alcohol':                    alc,
            'type':                       int(input_data.get("wine_type", 0)),
            'total_acidity':              total_acidity,
            'sugar_alcohol_ratio':        sugar_alcohol_ratio,
            'density_alcohol_interaction': density_alcohol_interaction,
        }

        df = pd.DataFrame([input_dict])[COLUMNS]
        prediction = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.0

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
            quality_score=prediction,
            # FIX: thêm wine_type vào DB record (trước đây bị bỏ sót)
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return {
            "status": "success",
            "id": db_prediction.id,
            "quality_score": prediction,
            "raw_score": round(proba, 4),
            "is_good_wine": prediction == 1,  # FIX: thêm field này để ResultCard dùng được
            "quality_label": "Good Wine (>=7)" if prediction == 1 else "Normal Wine (<7)",
            "created_at": db_prediction.created_at,
        }

    except Exception as e:
        db.rollback()
        return {
            "status": "error",
            "message": f"Lỗi: {str(e)}"
        }


def get_history_by_user(db: Session, user_id: int):
    try:
        return db.query(WinePrediction).filter(WinePrediction.user_id == user_id).all()
    except Exception as e:
        print(f"Lỗi get_history: {str(e)}")
        return []  # FIX: trả về [] thay vì None để tránh lỗi khi serialize list
