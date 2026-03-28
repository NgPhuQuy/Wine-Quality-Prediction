import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ======================
# 1. Load dataset
# ======================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = "red"
df_white["type"] = "white"

df = pd.concat([df_red, df_white], ignore_index=True)
df["type"] = df["type"].map({"red": 0, "white": 1})


df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]

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
wandb.init(project="wine-quality", name="SVM_Optimized")

# ======================
# 5. Pipeline
# ======================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# ======================
# 🔥 6. CV 5-FOLD (GIỐNG BẠN)
# ======================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(pipeline, X, y, cv=cv)

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.6f}")
    
    wandb.log({
        "cv_accuracy": score
    }, step=i)

wandb.log({
    "cv_mean_accuracy": cv_scores.mean(),
    "cv_std": cv_scores.std()
}, step=len(cv_scores))

# ======================
# 🔥 7. GridSearch tuning (TỐI ƯU SVM)
# ======================
param_grid = [
    {
        'svm__kernel': ['linear'],
        'svm__C': [0.1, 1, 10]
    },
    {
        'svm__kernel': ['rbf'],
        'svm__C': [1, 10, 100],
        'svm__gamma': ['scale', 0.1, 0.01]
    }
]

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

print("\n🚀 Đang chạy GridSearch SVM...")
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
# 11. Save model
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/svm_best.joblib")

wandb.finish()

print("\n✨ Done! Check WandB dashboard.")
