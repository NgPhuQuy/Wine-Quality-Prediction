import pandas as pd
import wandb
import joblib
import os
import numpy as np
from metrics import evaluate_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Data

df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = 0
df_white["type"] = 1 

df = pd.concat([df_red, df_white], ignore_index=True)

# 2. Feature Engineering
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]

# Chuyển chất lượng rượu thành nhị phân (1: Ngon [>=7], 0: Thường [<7])
df["quality"] = (df["quality"] >= 7).astype(int)

X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# 4. WandB Init
wandb.init(project="wine-quality", name="RF_CLASSIF_FINAL_V2")

# 5. Pipeline & Cross-Validation
# Thêm class_weight='balanced' vì tập dữ liệu Wine bị lệch nhãn
pipeline = Pipeline([('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--- Đang chạy Cross-Validation trên tập Train ---")
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    y_val_pred = pipeline.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred)
    cv_scores.append(f1_val)
    
    print(f"Fold {fold}: F1-Score = {f1_val:.4f}")
    wandb.log({"cv_fold_f1": f1_val})

# 6. Grid Search (Tối ưu hyperparameter)
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=kf, 
    scoring='f1', 
    n_jobs=-1,
    verbose=1
)

print("\n--- Đang chạy GridSearch ---")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 7. Đánh giá trên Test Set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test) # Lấy xác suất cả 2 lớp [P(0), P(1)]


evaluate_classification(y_test, y_pred, y_proba=y_proba, model_name="Random Forest Classification")

# Tính AUC riêng để log WandB
auc = roc_auc_score(y_test, y_proba[:, 1])

# 8. Feature Importance
feat_importance = pd.Series(
    best_model.named_steps['rf'].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Features:\n", feat_importance.head(10))

# Log kết quả cuối lên WandB
wandb.log({
    "best_cv_f1": grid_search.best_score_,
    "test_accuracy": accuracy_score(y_test, y_pred),
    "test_f1": f1_score(y_test, y_pred),
    "test_auc": auc,
    "best_params": str(grid_search.best_params_)
})

# 9. Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/rf_model.joblib")

wandb.finish()
print("\n--- HOÀN THÀNH PIPELINE ---")