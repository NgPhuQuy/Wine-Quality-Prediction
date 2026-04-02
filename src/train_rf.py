import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from metrics import evaluate_regression

df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = "red"
df_white["type"] = "white"

df = pd.concat([df_red, df_white], ignore_index=True)
df["type"] = df["type"].map({"red": 0, "white": 1})

df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]

df["quality"] = (df["quality"] >= 7).astype(int)

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

wandb.init(project="wine-quality", name="RF_CLASSIF_FINAL")

pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
f1_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)

    f1 = f1_score(y_val, y_pred)
    f1_list.append(f1)

    print(f"Fold {fold}: F1-Score = {f1:.6f}")
    wandb.log({"cv_f1_score": f1}, step=fold)

mean_f1 = sum(f1_list) / len(f1_list)
std_f1 = pd.Series(f1_list).std()

print(f"\n Mean F1: {mean_f1:.6f}")
print(f" Std F1: {std_f1:.6f}")

wandb.log({
    "cv_mean_f1": mean_f1,
    "cv_std_f1": std_f1
})

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
    cv=kf,
    scoring='f1',
    n_jobs=-1
)

print("\n Đang chạy GridSearch RF Classification...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1] 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n Best params: {grid_search.best_params_}")
print(f" Test RMSE: {rmse:.4f}")
print(f" Test MAE: {mae:.4f}")
print(f" R2 Score: {r2:.4f}")

feat_importance = pd.Series(
    best_model.named_steps['rf'].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n Top 10 Features:\n", feat_importance.head(10))

wandb.log({
    "best_cv_f1": grid_search.best_score_,
    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": recall,
    "test_f1": f1,
    "test_auc": auc
})

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/random_forest_classification.joblib")

wandb.finish()

print("\n DONE RF CLASSIFICATION PIPELINE!")