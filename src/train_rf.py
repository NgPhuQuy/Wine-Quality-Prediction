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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ==========================================
# 1. LOAD & PREP DATA
# ==========================================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")
df_red["type"] = 0
df_white["type"] = 1 
df = pd.concat([df_red, df_white], ignore_index=True)

# Feature Engineering
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
# 2. DANH SÁCH 6 CẤU HÌNH THỬ NGHIỆM
# ==========================================
configs = [
    {"name": "rf_classif_run_1", "params": {'rf__n_estimators': [500], 'rf__max_depth': [20], 'rf__min_samples_split': [2], 'rf__min_samples_leaf': [1], 'rf__max_features': ['sqrt'], 'rf__max_samples': [None]}},
    {"name": "rf_classif_run_2", "params": {'rf__n_estimators': [50], 'rf__max_depth': [3], 'rf__min_samples_split': [100], 'rf__min_samples_leaf': [50], 'rf__max_features': [0.1], 'rf__max_samples': [0.3]}},
    {"name": "rf_classif_run_3", "params": {'rf__n_estimators': [300], 'rf__max_depth': [10], 'rf__min_samples_split': [50], 'rf__min_samples_leaf': [30], 'rf__max_features': ['sqrt'], 'rf__max_samples': [0.6]}},
    {"name": "rf_classif_run_4", "params": {'rf__n_estimators': [400], 'rf__max_depth': [12], 'rf__min_samples_split': [20], 'rf__min_samples_leaf': [10], 'rf__max_features': [0.3], 'rf__max_samples': [0.8]}},
    {"name": "rf_classif_run_5", "params": {'rf__n_estimators': [300], 'rf__max_depth': [15], 'rf__min_samples_split': [30], 'rf__min_samples_leaf': [15], 'rf__max_features': [0.5], 'rf__max_samples': [0.7]}},
    {"name": "rf_classif_run_6", "params": {'rf__n_estimators': [200, 300], 'rf__max_depth': [8, 10, 12], 'rf__min_samples_split': [20, 40], 'rf__min_samples_leaf': [15, 25, 40], 'rf__max_features': [0.5, 'sqrt'], 'rf__max_samples': [0.7, 0.8]}}
]

best_f1 = -1.0
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
        
        print(f"\n>>> EXECUTING RUN: {run_cfg['name']}")

        pipeline = Pipeline([('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))])
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # GRID SEARCH
        grid_search = GridSearchCV(pipeline, run_cfg['params'], cv=kf, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # EVALUATION 
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)
        evaluate_classification(y_test, y_pred, y_proba=y_proba, model_name=run_cfg['name'])

        # LEARNING CURVE 
        plot_learning_curve(best_model, run_cfg['name'], X_train, y_train, kf)
       
        wandb.log({"learning_curve": wandb.Image(plt)}) 
        plt.close()

        # LOG METRICS & PLOTS
        current_f1 = f1_score(y_test, y_pred)
        current_auc = roc_auc_score(y_test, y_proba[:, 1])
        
        wandb.log({
            "best_cv_f1": grid_search.best_score_,
            "test_f1": current_f1,
            "test_auc": current_auc,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "roc_curve": wandb.plot.roc_curve(y_test, y_proba, labels=["Thường", "Ngon"]),
            "pr_curve": wandb.plot.pr_curve(y_test, y_proba, labels=["Thường", "Ngon"]),
            "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_test.values, preds=y_pred, class_names=["Thường", "Ngon"])
        })

        # SO SÁNH TÌM BEST
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_overall = best_model
            best_run_name = run_cfg['name']
            run.tags = run.tags + ("champion",)

        # 11. SAVE ARTIFACT 
        model_name = f"model_{run_cfg['name']}.joblib"
        joblib.dump(best_model, model_name)
        artifact = wandb.Artifact(run_cfg['name'], type='model')
        artifact.add_file(model_name)
        wandb.log_artifact(artifact)

# ==========================================
# 4. FINAL RUN ARCHIVE
# ==========================================
if best_model_overall:
    print(f"\nTHE BEST RUN: {best_run_name} with Test F1: {best_f1:.4f}")
    
    os.makedirs("models", exist_ok=True)
    champion_path = "models/rf_model.joblib"
    joblib.dump(best_model_overall, champion_path)

    with wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        name="rf_classif_final_champion",
        job_type="archive"
    ) as final_run:
        final_artifact = wandb.Artifact('champion-model', type='model', 
                                       description=f"Best model from: {best_run_name}")
        final_artifact.add_file(champion_path)
        final_run.log_artifact(final_artifact)
        final_run.log({"final_max_f1": best_f1})

print("\n--- PIPELINE COMPLETED SUCCESSFULLY ---")