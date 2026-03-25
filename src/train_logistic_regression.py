import pandas as pd
import wandb
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# 4. Khởi tạo wandb cho Logistic Regression
wandb.init(project="wine-quality", name="Logistic_Regression_Tuning")

# 5. Pipeline (Thêm StandardScaler là bắt buộc đối với Logistic Regression)
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

# 6. Hyperparameter tuning cho Logistic Regression
param_grid = {
    'lr__C': [0.1, 1.0, 10.0], # Tham số điều chuẩn (Regularization)
    'lr__solver': ['lbfgs', 'liblinear'] # Thuật toán tối ưu
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# 7. Dự đoán và kiểm tra
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# 8. Lưu model
os.makedirs("models", exist_ok=True)
joblib.dump(grid_search.best_estimator_, "models/logistic_regression_model.joblib")

# 9. Log kết quả lên wandb
wandb.log({
    "best_cv_accuracy": grid_search.best_score_,
    "test_accuracy": test_accuracy,
    "best_params": grid_search.best_params_
})

wandb.finish()