import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = "red"
df_white["type"] = "white"

df = pd.concat([df_red, df_white], ignore_index=True)
df["type"] = df["type"].map({"red": 0, "white": 1})


df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]


X = df.drop("quality", axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


wandb.init(project="wine-quality", name="RF_REGRESSION_FINAL")


pipeline = Pipeline([
    ('rf', RandomForestRegressor(random_state=42))
])


kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    rmse_list.append(rmse)

    print(f"Fold {fold}: RMSE = {rmse:.6f}")
    wandb.log({"cv_rmse": rmse}, step=fold)

mean_rmse = sum(rmse_list) / len(rmse_list)
std_rmse = pd.Series(rmse_list).std()

print(f"\n Mean RMSE: {mean_rmse:.6f}")
print(f" Std RMSE: {std_rmse:.6f}")

wandb.log({
    "cv_mean_rmse": mean_rmse,
    "cv_std": std_rmse
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
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

print("\n Đang chạy GridSearch RF...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

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
    "best_cv_rmse": -grid_search.best_score_,
    "test_rmse": rmse,
    "test_mae": mae,
    "test_r2": r2
})

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/random_forest_regression.joblib")

wandb.finish()

print("\n DONE RF REGRESSION PIPELINE!")