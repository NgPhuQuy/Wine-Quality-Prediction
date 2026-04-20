from . import model, COLUMNS
import pandas as pd

def predict_wine(data):
    try:
        # Chuyển data sang dạng dict nếu nó là object (để dùng .get cho an toàn)
        if hasattr(data, '__dict__'):
            data = data.__dict__
        else:
            data = data

        f_acid = float(data.get("fixed_acidity"))
        v_acid = float(data.get("volatile_acidity"))
        res_sugar = float(data.get("residual_sugar"))
        alc = float(data.get("alcohol"))
        dens = float(data.get("density"))
        
        total_acidity = f_acid + v_acid
        sugar_alcohol_ratio = res_sugar / (alc + 1e-5)
        density_alcohol_interaction = dens * alc

        input_dict = {
            'fixed acidity': f_acid,
            'volatile acidity': v_acid,
            'citric acid': float(data.get("citric_acid")),
            'residual sugar': res_sugar,
            'chlorides': float(data.get("chlorides")),
            'free sulfur dioxide': float(data.get("free_sulfur_dioxide")),
            'total sulfur dioxide': float(data.get("total_sulfur_dioxide")),
            'density': dens,
            'pH': float(data.get("pH")),
            'sulphates': float(data.get("sulphates")),
            'alcohol': alc,
            'type': int(data.get("type")),
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