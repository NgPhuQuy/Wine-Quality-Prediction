import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("data/winequality_combined.csv")

df["type"] = df["type"].map({
    "red": 0,
    "white": 1
})

# 2. Feature và target
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. wandb
wandb.init(project="wine-quality", name="RF_Tuning")

# 5. Pipeline
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))
])

# 6. Hyperparameter tuning
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# 7. Test
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# 8. Lưu model
os.makedirs("models", exist_ok=True)
joblib.dump(grid_search.best_estimator_, "models/random_forest_model.joblib")

# 9. Log wandb
wandb.log({
    "best_cv_accuracy": grid_search.best_score_,
    "test_accuracy": test_accuracy,
    "best_params": grid_search.best_params_
})

wandb.finish()