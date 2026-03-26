import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ======================
# 1. Load dataset
# ======================
df = pd.read_csv("data/winequality_combined.csv")

df["type"] = df["type"].map({"red": 0, "white": 1})

# ======================
# 2. Feature & target
# ======================
X = df.drop("quality", axis=1)
y = (df["quality"] >= 6).astype(int)

# ======================
# 3. Train/Test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# ======================
# 4. WandB init
# ======================
wandb.init(project="wine-quality", name="Logistic_Regression_Optimized")

# ======================
# 5. Pipeline
# ======================
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('lr', LogisticRegression(max_iter=5000, random_state=42))
])

# ======================
# 🔥 6. CV để so sánh (GIỐNG NOTEBOOK)
# ======================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(pipeline, X, y, cv=cv)

# 👉 log từng fold với step rõ ràng
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.6f}")
    
    wandb.log({
        "cv_accuracy": score
    }, step=i)   # 🔥 QUAN TRỌNG: set step

# 👉 log tổng (ở step cuối)
wandb.log({
    "cv_mean_accuracy": cv_scores.mean(),
    "cv_std": cv_scores.std()
}, step=len(cv_scores))

# ======================
# 7. GridSearch tuning
# ======================
param_grid = {
    'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'lr__solver': ['liblinear', 'lbfgs'],
    'lr__class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='f1',
    n_jobs=-1
)

print("\n🚀 Đang chạy GridSearch...")
grid_search.fit(X_train, y_train)

# ======================
# 8. Best model
# ======================
best_model = grid_search.best_estimator_

# ======================
# 9. Evaluate
# ======================
y_pred = best_model.predict(X_test)
y_probas = best_model.predict_proba(X_test)

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\n✅ Best params: {grid_search.best_params_}")
print(f"🎯 Test Accuracy: {test_acc:.4f}")
print(f"📊 Test F1: {test_f1:.4f}")
print("\n📄 Report:\n", classification_report(y_test, y_pred))

# ======================
# 10. Log wandb
# ======================
wandb.log({
    
    "best_cv_f1": grid_search.best_score_,
    "test_accuracy": test_acc,
    "test_f1": test_f1,
    "best_C": grid_search.best_params_['lr__C'],
    "best_solver": grid_search.best_params_['lr__solver'],
    "class_weight": str(grid_search.best_params_['lr__class_weight'])
})

# (optional) confusion matrix đẹp
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test.values,   # 🔥 FIX
        preds=y_pred            # cái này đã là numpy rồi
    )
})


# ======================
# 11. Save model
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/logistic_regression_best.joblib")

wandb.finish()

print("\n✨ Done! Check WandB dashboard.")