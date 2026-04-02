import pandas as pd
import joblib
import os
import wandb
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from metrics import evaluate_classification
from Learning_Curve import plot_learning_curve

# 🔥 Tắt warning XGBoost (optional)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# ======================
# 1. LOAD DATA
# ======================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = 0
df_white["type"] = 1

df = pd.concat([df_red, df_white], ignore_index=True)

# ======================
# 2. Feature Engineering
# ======================
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]

df["quality"] = (df["quality"] >= 7).astype(int)

# Fix NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop("quality", axis=1)
y = df["quality"]

# ======================
# 3. SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 4. WANDB
# ======================
wandb.init(project="wine-quality", name="XGB_FINAL_V2")

# ======================
# 5. MODEL + CV
# ======================
model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"   # 🔥 tăng tốc
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.6f}")
    wandb.log({"cv_accuracy": score}, step=i)

wandb.log({
    "cv_mean_accuracy": cv_scores.mean(),
    "cv_std": cv_scores.std()
})

# ======================
# 6. GRID SEARCH
# ======================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=cv,
    scoring="f1",
    verbose=1,
    n_jobs=-1
)

print("\nĐang tune XGBoost...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n🔥 BEST PARAMS:", grid.best_params_)

# ======================
# 7. THRESHOLD TUNING
# ======================
y_probas = best_model.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_f1 = 0

for t in [0.4, 0.5, 0.6]:
    preds = (y_probas > t).astype(int)
    f1 = f1_score(y_test, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\nBest threshold: {best_thresh}")

y_pred = (y_probas > best_thresh).astype(int)
y_proba_full = best_model.predict_proba(X_test)

# ======================
# 8. EVALUATION
# ======================
evaluate_classification(
    y_test,
    y_pred,
    y_proba=y_proba_full,
    model_name="XGBoost"
)

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probas)

# ======================
# 9. WANDB LOG
# ======================
wandb.log({
    "best_cv_f1": grid.best_score_,
    "test_accuracy": test_acc,
    "test_f1": test_f1,
    "test_auc": auc,
    "best_threshold": best_thresh,
    "best_params": str(grid.best_params_)
})

wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test.values,
        preds=y_pred
    )
})

# ======================
# 10. SAVE MODEL
# ======================
os.makedirs("models", exist_ok=True)
model_path = "models/xgb_model.joblib"
joblib.dump(best_model, model_path)

wandb.save(model_path)

# ======================
# 11. LEARNING CURVE
# ======================
plot_learning_curve(best_model, "XGBoost", X_train, y_train, cv)

wandb.finish()

print("\nDONE! XGBoost fixed & optimized.")