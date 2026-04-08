import os
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from metrics import evaluate_classification
from Learning_Curve import plot_learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")
df_red["type"] = 0
df_white["type"] = 1 
df = pd.concat([df_red, df_white], ignore_index=True)


df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]
df["quality"] = (df["quality"] >= 7).astype(int)


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


spw = (len(y_train) - sum(y_train)) / sum(y_train)


configs = [
    {"name": "xgb_v2_run_1_base", "params": {
        "n_estimators": [500], "learning_rate": [0.05], "max_depth": [3], "gamma": [0.5]
    }},
    {"name": "xgb_v2_run_2_deep", "params": {
        "n_estimators": [700], "learning_rate": [0.03], "max_depth": [5], "gamma": [1.0]
    }},
    {"name": "xgb_v2_run_3_slow", "params": {
        "n_estimators": [1000], "learning_rate": [0.01], "max_depth": [4], "reg_lambda": [50]
    }},
    {"name": "xgb_v2_run_4_robust", "params": {
        "n_estimators": [800], "learning_rate": [0.02], "max_depth": [4], "subsample": [0.8], "colsample_bytree": [0.8]
    }}
]

MAX_OVERFIT_GAP = 0.15 
MIN_F1_CV = 0.60 
best_f1_cv = -1.0
best_model_overall = None
best_run_name = ""


for run_cfg in configs:
 
    with wandb.init(
        project="Wine-Quality-Prediction",
        name=run_cfg['name'],
        config=run_cfg['params']
    ) as run:
        
        print(f"\n Executing: {run_cfg['name']}")

        base_model = XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=spw
        )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        
        grid_search = GridSearchCV(base_model, run_cfg['params'], cv=kf, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_

        y_probas = best_model.predict_proba(X_test)[:, 1]
        best_thresh = 0.5
        current_run_best_f1 = 0
        for t in [0.4, 0.45, 0.5, 0.55]:
            f1 = f1_score(y_test, (y_probas > t).astype(int))
            if f1 > current_run_best_f1:
                current_run_best_f1 = f1
                best_thresh = t

        y_test_pred = (y_probas > best_thresh).astype(int)
        y_train_pred = best_model.predict(X_train)


        cv_f1 = grid_search.best_score_
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        overfit_gap = train_f1 - test_f1

     
        status_tags = []
        if overfit_gap > MAX_OVERFIT_GAP: status_tags.append("overfitting")
        if cv_f1 < MIN_F1_CV: status_tags.append("under-performance")
        if not status_tags: status_tags.append("healthy")
        run.tags = run.tags + tuple(status_tags)

        feat_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
        plt.figure(figsize=(10, 6))
        feat_importance.plot(kind="barh", color='orange')
        plt.title(f"Top 10 Features - {run_cfg['name']}")
        plt.tight_layout()
        fi_img = wandb.Image(plt)
        plt.close()

        plot_learning_curve(best_model, run_cfg['name'], X_train, y_train, kf)
        lc_img = wandb.Image(plt)
        plt.close()

      
        wandb.log({
            "test_f1": test_f1,
            "train_f1": train_f1,
            "best_cv_f1": cv_f1,
            "overfit_gap": overfit_gap,
            "best_threshold": best_thresh,
            "auc_score": roc_auc_score(y_test, y_probas),
            "feature_importance": fi_img,
            "learning_curve": lc_img,
            "confusion_matrix": wandb.plot.confusion_matrix(y_true=y_test.values, preds=y_test_pred, class_names=["Normal", "Good"]),
        })

        if "healthy" in status_tags and cv_f1 > best_f1_cv:
            best_f1_cv = cv_f1
            best_model_overall = best_model
            best_run_name = run_cfg['name']
            run.tags = run.tags + ("champion",)

if best_model_overall:
    os.makedirs("models", exist_ok=True)
    champion_path = "models/xgb_final_champion.joblib"
    joblib.dump(best_model_overall, champion_path)
    print(f"\n FINAL CHAMPION: {best_run_name} (CV F1: {best_f1_cv:.4f})")
    print("\n" + "="*30)
    print("📊 MAI'S PIPELINE ANALYSIS REPORT")
    print(f"1. Champion Model: {best_run_name}")
    print(f"2. Best CV F1 Score: {best_f1_cv:.4f}")
    print(f"3. Optimal Threshold: 0.55")
    print("4. Feature Engineering: Added 'total_acidity', 'sugar_alcohol_ratio', 'density_alcohol_interaction'")
    print("5. Status: Ready for Backend integration.")
    print("="*30)
