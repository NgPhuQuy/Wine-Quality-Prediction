# import joblib
# import pandas as pd

# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")

# columns = [
#     'fixed acidity',
#     'volatile acidity',
#     'citric acid',
#     'residual sugar',
#     'chlorides',
#     'free sulfur dioxide',
#     'total sulfur dioxide',
#     'density',
#     'pH',
#     'sulphates',
#     'alcohol',
#     'type'
# ]

def predict_wine(data):
    # input_dict = {
    #     "fixed_acidity": data.fixed_acidity,
    #     "volatile_acidity": data.volatile_acidity,
    #     "citric_acid": data.citric_acid,
    #     "residual_sugar": data.residual_sugar,
    #     "chlorides": data.chlorides,
    #     "free_sulfur_dioxide": data.free_sulfur_dioxide,
    #     "total_sulfur_dioxide": data.total_sulfur_dioxide,
    #     "density": data.density,
    #     "pH": data.pH,
    #     "sulphates": data.sulphates,
    #     "alcohol": data.alcohol,
    #     "type": 0 if data.type == "red" else 1
    # }

    # df = pd.DataFrame([input_dict])[columns]  # 🔥 ép đúng thứ tự

    # X_scaled = scaler.transform(df)
    # pred = model.predict(X_scaled)[0]

    return {
        "quality": 7,
        "raw_score": 2
    }