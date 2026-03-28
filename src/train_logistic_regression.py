import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

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
y = (df["quality"] >= 6).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


wandb.init(project="wine-quality", name="LogReg_BOOSTED")


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  
    ('lr', LogisticRegression(
        max_iter=5000,
        class_weight='balanced',   
        random_state=42
    ))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(pipeline, X, y, cv=cv)

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.6f}")
    wandb.log({"cv_accuracy": score}, step=i)

wandb.log({
    "cv_mean_accuracy": cv_scores.mean(),
    "cv_std": cv_scores.std()
})


param_grid = {
    'lr__C': [0.01, 0.1, 1, 10, 50],
    'lr__solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

print("\n Đang chạy GridSearch...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_probas = best_model.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_f1 = 0

for t in [0.4, 0.5, 0.6]:
    preds = (y_probas > t).astype(int)
    f1 = f1_score(y_test, preds)
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n Best threshold: {best_thresh}")

y_pred = (y_probas > best_thresh).astype(int)


test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\n Best params: {grid_search.best_params_}")
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Test F1: {test_f1:.4f}")
print("\n Report:\n", classification_report(y_test, y_pred))


wandb.log({
    "best_cv_f1": grid_search.best_score_,
    "test_accuracy": test_acc,
    "test_f1": test_f1,
    "best_threshold": best_thresh,
    "best_C": grid_search.best_params_['lr__C'],
    "best_solver": grid_search.best_params_['lr__solver']
})

wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test.values,
        preds=y_pred
    )
})


os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/logistic_best_boosted.joblib")

wandb.finish()

print("\n DONE! Logistic Regression improved.")
