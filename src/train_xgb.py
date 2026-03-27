import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ======================
# 1. LOAD DATA
# ======================
df = pd.read_csv("../data/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"] - 3  # fix label

# ======================
# 2. SPLIT (🔥 STRATIFY)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 3. BASE MODEL
# ======================
xgb = XGBClassifier(
    eval_metric="mlogloss"
)

# ======================
# 4. GRID SEARCH (🔥 TUNE)
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
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)

print("tune model...")
grid.fit(X_train, y_train)

# ======================
# 5. BEST MODEL
# ======================
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

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ======================
# 7. FEATURE IMPORTANCE
# ======================
importance = best_model.feature_importances_

print("\n⭐ Feature Importance:")
for i, col in enumerate(X.columns):
    print(f"{col}: {importance[i]:.4f}")

# ======================
# 8. SAVE MODEL
# ======================
joblib.dump(best_model, "../models/xgboost_best.joblib")

print("\n✅ Saved: xgboost_best.joblib")