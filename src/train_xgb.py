import pandas as pd
import joblib
import os
import wandb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ======================
# 0. INIT WANDB
# ======================
wandb.init(
    project="wine-quality-xgboost",
    name="xgb-gridsearch",
    config={
        "model": "XGBoost",
        "test_size": 0.2,
        "cv": 5
    }
)

# ======================
# 1. LOAD DATA
# ======================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = "red"
df_white["type"] = "white"

df = pd.concat([df_red, df_white], ignore_index=True)
df["type"] = df["type"].map({"red": 0, "white": 1})

# ======================
# 2. Feature & target
# ======================
X = df.drop("quality", axis=1)
y = (df["quality"] >= 6).astype(int)

# ======================
# 3. SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 4. MODEL
# ======================
xgb = XGBClassifier(eval_metric="mlogloss")

# ======================
# 5. GRID SEARCH
# ======================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1]
}

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)

print("tune model...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n🔥 BEST PARAMS:", grid.best_params_)

# ======================
# 6. EVALUATION
# ======================
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n🔥 FINAL RESULT 🔥")
print("Accuracy:", acc)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\n📉 Confusion Matrix:")
print(cm)

# ======================
# 7. WANDB LOG
# ======================
wandb.log({
    "accuracy": acc,
    "best_params": grid.best_params_
})

# log confusion matrix (🔥 hay dùng)
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test,
        preds=y_pred
    )
})

# ======================
# 8. SAVE MODEL
# ======================
os.makedirs("models", exist_ok=True)
model_path = "models/xgboost_best.joblib"
joblib.dump(best_model, model_path)

# log model lên wandb luôn
wandb.save(model_path)

print("\n✅ Saved:", model_path)

wandb.finish()