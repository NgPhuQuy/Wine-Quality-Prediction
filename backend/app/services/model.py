import joblib
import numpy as np
import pandas as pd
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..", "data", "winequality-red.csv")
)

XGB_PATH = os.path.join(CURRENT_DIR, "xgboost.pkl")
LGBM_PATH = os.path.join(CURRENT_DIR, "lightgbm.pkl")


def train_models():
    print("👉 Đang đọc file:", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"] - 3  

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train)


    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)


    xgb_pred = xgb.predict(X_test)
    lgbm_pred = lgbm.predict(X_test)

    print(" XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
    print(" LightGBM Accuracy:", accuracy_score(y_test, lgbm_pred))

    joblib.dump(xgb, XGB_PATH)
    joblib.dump(lgbm, LGBM_PATH)

    print(" Train xong và đã lưu model")

    return xgb, lgbm


def load_models():
    if os.path.exists(XGB_PATH) and os.path.exists(LGBM_PATH):
        print(" Load model có sẵn")
        xgb = joblib.load(XGB_PATH)
        lgbm = joblib.load(LGBM_PATH)
    else:
        print(" Chưa có model → train mới")
        xgb, lgbm = train_models()

    return xgb, lgbm

xgb_model, lgbm_model = load_models()


def predict_wine(data, model_name="xgboost"):
    start = time.time()  

    features = pd.DataFrame([{
        "fixed_acidity": data.fixed_acidity,
        "volatile_acidity": data.volatile_acidity,
        "citric_acid": data.citric_acid,
        "residual_sugar": data.residual_sugar,
        "chlorides": data.chlorides,
        "free_sulfur_dioxide": data.free_sulfur_dioxide,
        "total_sulfur_dioxide": data.total_sulfur_dioxide,
        "density": data.density,
        "pH": data.pH,
        "sulphates": data.sulphates,
        "alcohol": data.alcohol
    }])

    if model_name == "lightgbm":
        prediction = lgbm_model.predict(features)
    else:
        prediction = xgb_model.predict(features)

    end = time.time() 

    return {
        "quality": int(prediction[0] + 3),
        "time": round(end - start, 4) 
    }