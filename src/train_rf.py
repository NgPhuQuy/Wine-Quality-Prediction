import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ======================
# 1. Load dataset
# ======================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = "red"
df_white["type"] = "white"

df = pd.concat([df_red, df_white], ignore_index=True)
df["type"] = df["type"].map({"red": 0, "white": 1})

# ======================
# 2. Feature Engineering
# ======================
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]



# ======================
# 3. Feature & target
# ======================
X = df.drop("quality", axis=1)
y = (df["quality"] >= 6).astype(int)

# ======================
# 4. Train/Test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================
# 5. WandB init
# ======================
wandb.init(project="wine-quality", name="RF_FULL_FINAL")

# ======================
# 6. Pipeline
# ======================
pipeline = Pipeline([
    ('rf', RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

# ======================
# 🔥 7. Cross Validation (5 fold)
# ======================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.6f}")
    wandb.log({"cv_f1": score}, step=i)

wandb.log({
    "cv_mean_f1": cv_scores.mean(),
    "cv_std": cv_scores.std()
}, step=len(cv_scores))

# ======================
# 🔥 8. GridSearch
# ======================
param_grid = {
    'rf__n_estimators': [200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2],
    'rf__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

print("\n🚀 Đang chạy GridSearch RF...")
grid_search.fit(X_train, y_train)

# ======================
# 9. Evaluate
# ======================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\n✅ Best params: {grid_search.best_params_}")
print(f"🎯 Test Accuracy: {test_acc:.4f}")
print(f"📊 Test F1: {test_f1:.4f}")
print("\n📄 Report:\n", classification_report(y_test, y_pred))

# ======================
# 🔥 10. Feature Importance
# ======================
feat_importance = pd.Series(
    best_model.named_steps['rf'].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n🔥 Top 10 Features:\n", feat_importance.head(10))

# ======================
# 11. Log wandb
# ======================
wandb.log({
    "best_cv_f1": grid_search.best_score_,
    "test_accuracy": test_acc,
    "test_f1": test_f1
})

wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test.values,
        preds=y_pred
    )
})

# ======================
# 12. Save model
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/random_forest_full.joblib")

wandb.finish()

print("\n🔥 DONE RF FULL PIPELINE!")
