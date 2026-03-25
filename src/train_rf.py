import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Khởi tạo wandb
wandb.init(project="wine-quality", name="RF_Tuning")

# 2. Tạo Pipeline 
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# 3. Hyperparameter Tuning 
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 4. Log kết quả tốt nhất lên wandb 
wandb.log({"best_accuracy": grid_search.best_score_})
wandb.finish()