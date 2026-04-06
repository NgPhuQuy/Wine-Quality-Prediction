import pandas as pd
import wandb
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from metrics import evaluate_classification
from Learning_Curve import plot_learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# ==========================================
# 1. LOAD & PREPROCESS DATA
# ==========================================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")
df_red["type"] = 0
df_white["type"] = 1 
df = pd.concat([df_red, df_white], ignore_index=True)

df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]
df["quality"] = (df["quality"] >= 7).astype(int)

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================
# 2. CONFIGURATIONS & HYPERPARAMETERS
# ==========================================
configs = [
    {"name": "rf_classif_run_1", "params": {
        'rf__n_estimators': [800], 'rf__max_depth': [11], 'rf__min_samples_split': [30], 
        'rf__min_samples_leaf': [15], 'rf__max_features': [0.2], 'rf__max_samples': [0.75]
    }},
    {"name": "rf_classif_run_2", "params": {
        'rf__n_estimators': [900], 'rf__max_depth': [12], 'rf__min_samples_split': [25], 
        'rf__min_samples_leaf': [14], 'rf__max_features': [0.22], 'rf__max_samples': [0.8]
    }},
    {"name": "rf_classif_run_3", "params": {
        'rf__n_estimators': [1000], 'rf__max_depth': [10], 'rf__min_samples_split': [35], 
        'rf__min_samples_leaf': [16], 'rf__max_features': ['sqrt'], 'rf__max_samples': [0.7]
    }},
    {"name": "rf_classif_run_4", "params": {
        'rf__n_estimators': [1100], 'rf__max_depth': [11], 'rf__min_samples_split': [28], 
        'rf__min_samples_leaf': [15], 'rf__max_features': [0.25], 'rf__max_samples': [0.75]
    }},
    {"name": "rf_classif_run_5", "params": {
        'rf__n_estimators': [950], 'rf__max_depth': [12], 'rf__min_samples_split': [22], 
        'rf__min_samples_leaf': [13], 'rf__max_features': [0.28], 'rf__max_samples': [0.8]
    }},
    {"name": "rf_classif_run_6", "params": {
        'rf__n_estimators': [1200], 'rf__max_depth': [11], 'rf__min_samples_split': [26], 
        'rf__min_samples_leaf': [14], 'rf__max_features': [0.25], 'rf__max_samples': [0.78]
    }},
    {"name": "rf_classif_run_7", "params": {
        'rf__n_estimators': [1300], 'rf__max_depth': [10], 'rf__min_samples_split': [40], 
        'rf__min_samples_leaf': [18], 'rf__max_features': [0.3], 'rf__max_samples': [0.7]
    }},
    {"name": "rf_classif_run_8", "params": {
        'rf__n_estimators': [1500], 'rf__max_depth': [12], 'rf__min_samples_split': [20], 
        'rf__min_samples_leaf': [12], 'rf__max_features': [0.2], 'rf__max_samples': [0.85]
    }}
]

MAX_OVERFIT_GAP = 0.1 
MIN_F1_CV = 0.60 
best_f1_cv = -1.0
best_model_overall = None
best_run_name = ""

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for run_cfg in configs:
    with wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        name=run_cfg['name'],
        config=run_cfg['params']
    ) as run:
        
        print(f"Executing: {run_cfg['name']}")

        pipeline = Pipeline([('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))])
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(pipeline, run_cfg['params'], cv=kf, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)
        y_train_pred = best_model.predict(X_train)

        evaluate_classification(y_test, y_test_pred, y_proba=y_test_proba, model_name=run_cfg['name'])

        cv_f1 = grid_search.best_score_
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        overfit_gap = train_f1 - test_f1

        status_tags = []
        if overfit_gap > MAX_OVERFIT_GAP: status_tags.append("overfitting")
        if cv_f1 < MIN_F1_CV: status_tags.append("underfitting")
        if not status_tags: status_tags.append("healthy")
        run.tags = run.tags + tuple(status_tags)

        plot_learning_curve(best_model, run_cfg['name'], X_train, y_train, kf)
        lc_img = wandb.Image(plt)
        plt.close()

        importances = best_model.named_steps['rf'].feature_importances_
        features = X.columns
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.tight_layout()
        fi_img = wandb.Image(plt)
        plt.close()

        wandb.log({
            "learning_curve": lc_img,
            "feature_importance": fi_img,
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_test.values, preds=y_test_pred, class_names=["Normal", "Good"]),
            "pr_curve": wandb.plot.pr_curve(y_true=y_test.values, y_probas=y_test_proba, labels=["Normal", "Good"]),
            "best_cv_f1": cv_f1,
            "overfit_gap": overfit_gap,
            "test_f1": test_f1,
            "train_f1": train_f1
        })

        if "healthy" in status_tags and cv_f1 > best_f1_cv:
            best_f1_cv = cv_f1
            best_model_overall = best_model
            best_run_name = run_cfg['name']
            run.tags = run.tags + ("champion",)
            print(f"New candidate champion found: {best_run_name} with CV F1: {cv_f1:.4f}")

        temp_name = f"temp_{run_cfg['name']}.joblib"
        joblib.dump(best_model, temp_name)
        artifact = wandb.Artifact(run_cfg['name'], type='model')
        artifact.add_file(temp_name)
        wandb.log_artifact(artifact)
        os.remove(temp_name)

# ==========================================
# 4. FINAL CHAMPION ARCHIVE & EXPORT
# ==========================================
if best_model_overall:
    print("-" * 30)
    print(f"FINAL SELECTION: {best_run_name}")
    print(f"BEST CV F1 SCORE: {best_f1_cv:.4f}")
    print("-" * 30)
    
    os.makedirs("models", exist_ok=True)
    champion_path = "models/rf_model.joblib"
    joblib.dump(best_model_overall, champion_path)
    
    with wandb.init(project="Wine-Quality-Prediction", name="rf_final_champion", job_type="archive") as final_run:
        final_artifact = wandb.Artifact('champion-model', type='model')
        final_artifact.add_file(champion_path)
        final_run.log_artifact(final_artifact)
        final_run.log({"final_best_cv_f1": best_f1_cv})
else:
    print("No healthy models met the minimum CV F1 threshold.")