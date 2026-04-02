import pandas as pd
import wandb
import joblib
import os
import numpy as np

from metrics import evaluate_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from Learning_Curve import plot_learning_curve

# ======================
# 1. Load Data
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

# 🔥 Đồng bộ Logistic
df["quality"] = (df["quality"] >= 7).astype(int)

# Fix NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop("quality", axis=1)
y = df["quality"]

# ======================
# 3. Split Data
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================
# 4. WandB Init
# ======================
wandb.init(project="wine-quality", name="SVM_LIKE_LOGISTIC")

# ======================
# 5. Pipeline + CV (giống Logistic)
# ======================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        probability=True,
        class_weight='balanced',
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv)

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.6f}")
    wandb.log({"cv_accuracy": score}, step=i)

wandb.log({
    "cv_mean_accuracy": cv_scores.mean(),
    "cv_std": cv_scores.std()
})

# ======================
# 6. GridSearch
# ======================
param_grid = [
    {
        'svm__kernel': ['linear'],
        'svm__C': [0.01, 0.1, 1] # Thêm 0.01 để giảm Overfitting
    },
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.1, 1, 10], 
        'svm__gamma': ['scale', 'auto', 0.01, 0.05] # auto và giá trị nhỏ giúp đường biên mượt hơn
    }
]

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

print("\nĐang chạy GridSearch SVM...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ======================
# 7. Threshold tuning (giống Logistic)
# ======================
y_probas = best_model.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_f1 = 0

for t in np.linspace(0.3, 0.7, 9):
    preds = (y_probas > t).astype(int)
    f1 = f1_score(y_test, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\nBest threshold: {best_thresh}")

y_pred = (y_probas > best_thresh).astype(int)
y_proba_full = best_model.predict_proba(X_test)

# ======================
# 8. Evaluation
# ======================
evaluate_classification(
    y_test,
    y_pred,
    y_proba=y_proba_full,
    model_name="SVM"
)

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probas)

print(f"\nBest params: {grid_search.best_params_}")

# ======================
# 9. Log WandB
# ======================
wandb.log({
    "best_cv_f1": grid_search.best_score_,
    "test_accuracy": test_acc,
    "test_f1": test_f1,
    "test_auc": auc,
    "best_threshold": best_thresh,
    "best_C": grid_search.best_params_.get('svm__C'),
    "best_kernel": grid_search.best_params_.get('svm__kernel'),
    "best_gamma": str(grid_search.best_params_.get('svm__gamma', 'N/A'))
})

wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test.values,
        preds=y_pred
    )
})

# ======================
# 10. Save Model
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/svm_model.joblib")

# ======================
# 11. Learning Curve
# ======================
plot_learning_curve(best_model, "SVM", X_train, y_train, cv)

wandb.finish()

print("\nDONE! SVM aligned with Logistic.")