import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from Learning_Curve import plot_learning_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from metrics import evaluate_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

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

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 2. CONFIGURATIONS & HYPERPARAMETERS
# ==========================================
# ==========================================
# 2. CONFIGURATIONS & HYPERPARAMETERS (BỘ 7 TINH TÚY)
# ==========================================
configs = [
    # 1. Quán quân cân bằng \
    {
        "name": "lg_run1", 
        "metric": "f1",
        "class_weight": None,
        "smote": False,
        "threshold": 0.25,
        "penalty": "l2",
    },
    # 2. Quán quân bao phủ 
    {
        "name": "lg_run2", 
        "metric": "recall",
        "class_weight": {0: 1, 1: 3.5},
        "smote": False,
        "threshold": 0.5,
        "penalty": "l2",
    },
    # 3. Quán quân SMOTE )
    {
        "name": "lg_run3", 
        "metric": "f1",
        "class_weight": None,
        "smote": True,
        "threshold": 0.6,
        "penalty": "l2",
    },
    # 4. Quán quân lọc nhiễu 
    {
        "name": "lg_run4", 
        "metric": "f1",
        "class_weight": "balanced",
        "smote": False,
        "threshold": 0.6,
        "penalty": "l1",
    },
    
    
    # 7. Quán quân bắn tỉa cực đoan 
    {
        "name": "lg_run5", 
        "metric": "f1",
        "class_weight": {0: 1, 1: 3},
        "smote": False,
        "threshold": 0.75,
        "penalty": "l2",
    },
    {
        "name": "lg_run6", 
        "metric": "f1",
        "class_weight": "balanced", 
        "smote": True,              
        "threshold": 0.65,          
        "penalty": "l2",
    },
    {
        "name": "lg_run7", 
        "metric": "f1",
        "class_weight": "balanced", 
        "smote": True, 
        "threshold": 0.7,           
        "penalty": "l1",            
    }
]
MAX_OVERFIT_GAP = 0.1
MIN_F1_CV = 0.4
best_f1_cv = -1.0
best_model_overall = None
best_run_name = ""

stability_history = []
os.makedirs("models", exist_ok=True)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for config in configs:
    with wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        name=config["name"],
        tags=["logistic", config["metric"]],
    ) as run:

        print(f"\nExecuting: {config['name']}")

        # Khởi tạo Pipeline
        steps = [("scaler", StandardScaler())]
        if config["smote"]:
            steps.append(("smote", SMOTE(random_state=42)))

        lr_params = {
            "max_iter": 5000,
            "class_weight": config["class_weight"],
            "random_state": 42,
        }

        if config["penalty"] == "l1":
            lr_params["solver"] = "liblinear"
        else:
            lr_params["solver"] = "lbfgs"
            lr_params["l1_ratio"] = 0

        steps.append(("lr", LogisticRegression(**lr_params)))
        pipeline = ImbPipeline(steps)

        # ------------------------------------------
        # Cross Validation (Tracking AUC & F1)
        # ------------------------------------------
        cv_auc_scores = []
        cv_f1_scores = [] # Thêm để lưu F1 từng fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_val_proba = pipeline.predict_proba(X_val)[:, 1]
            cv_auc_scores.append(roc_auc_score(y_val, y_val_proba))
            
            # Tính F1 cho fold hiện tại dựa trên threshold của config
            y_val_pred_fold = (y_val_proba >= config["threshold"]).astype(int)
            cv_f1_scores.append(f1_score(y_val, y_val_pred_fold))

        mean_cv_auc = np.mean(cv_auc_scores)
        std_cv_auc = np.std(cv_auc_scores)
        mean_cv_f1 = np.mean(cv_f1_scores) # Tính trung bình F1 từ CV

        # ------------------------------------------
        # GridSearch
        # ------------------------------------------
        param_grid = {"lr__C": [0.01, 0.1, 1, 10]}
        grid_scorer = make_scorer(
            lambda y_true, y_proba, threshold=config["threshold"]: 
                recall_score(y_true, (y_proba >= threshold).astype(int)),
            response_method="predict_proba"
        )
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=kf,
            scoring=config["metric"],
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # ------------------------------------------
        # Test Evaluation & Tính toán thông số
        # ------------------------------------------
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= config["threshold"]).astype(int)

        test_metric_score = recall_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)
        overfit_gap = abs(mean_cv_auc - test_auc)

        t_acc = round(accuracy_score(y_test, y_pred), 4)
        t_f1 = round(f1_score(y_test, y_pred), 4)
        t_prec = round(precision_score(y_test, y_pred), 4)
        t_rec = round(recall_score(y_test, y_pred), 4)

        evaluate_classification(
            y_test,
            y_pred,
            y_proba=best_model.predict_proba(X_test),
            model_name=config["name"],
        )

        stability_history.append(
            {
                "name": config["name"],
                "cv_f1": round(mean_cv_f1, 4), # Thêm vào để theo dõi
                "mean_cv_auc": round(mean_cv_auc, 4),
                "test_auc": round(test_auc, 4),
                "overfit_gap": round(overfit_gap, 4),
                "cv_std": round(std_cv_auc, 4),
                "acc": t_acc,
                "f1": t_f1,
                "prec": t_prec,
                "rec": t_rec,
            }
        )

        # ------------------------------------------
        # Gắn Tag tự động (theo chuẩn RF)
        # ------------------------------------------
        status_tags = []
        if overfit_gap > MAX_OVERFIT_GAP:
            status_tags.append("overfitting")
        if mean_cv_f1 < MIN_F1_CV:
            status_tags.append("underfitting")
        if not status_tags:
            status_tags.append("healthy")
        run.tags = run.tags + tuple(status_tags)

        # 1. Log Learning Curve
        plot_learning_curve(best_model, config["name"], X_train, y_train, kf)
        lc_img = wandb.Image(plt)
        plt.close()

        # 2. Xử lý Top 10 Feature Importance (Hệ số Coefficients)
        importances = best_model.named_steps["lr"].coef_[0]
        features = X.columns
        indices = np.argsort(np.abs(importances))[-10:]

        plt.figure(figsize=(10, 6))
        plt.barh(
            range(len(indices)),
            importances[indices],
            color="skyblue",
            align="center",
        )
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
        plt.title(f"Top 10 Feature Importance - {config['name']}")
        fi_img = wandb.Image(plt)
        plt.close()

        fi_table = wandb.Table(
            columns=["Feature", "Importance"],
            data=[[features[i], importances[i]] for i in reversed(indices)],
        )

        # 3. Tổng hợp log lên WandB
        wandb.log(
            {
                "learning_curve": lc_img,
                "feature_importance_image": fi_img,
                "top_10_features_chart": wandb.plot.bar(
                    fi_table, "Feature", "Importance", title="Top 10 Features"
                ),
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_test.values,
                    preds=y_pred,
                    class_names=["Normal", "Good"],
                ),
                "pr_curve": wandb.plot.pr_curve(
                    y_true=y_test.values,
                    y_probas=best_model.predict_proba(X_test),
                    labels=["Normal", "Good"],
                ),
                "mean_cv_auc": mean_cv_auc,
                "mean_cv_f1": mean_cv_f1, # Log thêm CV F1
                "cv_std_auc": std_cv_auc,
                "overfit_gap": overfit_gap,
                "test_auc": test_auc,
                "test_f1": t_f1,
                "test_precision": t_prec,
                "test_recall": t_rec,
                "test_accuracy": t_acc,
            }
        )

        # Cập nhật ứng cử viên Champion (Dựa trên CV F1 cao nhất và "healthy")
        if "healthy" in status_tags and mean_cv_f1 > best_f1_cv:
            best_f1_cv = mean_cv_f1
            best_model_overall = best_model
            best_run_name = config["name"]
            run.tags = run.tags + ("champion",)
            print(
                f"New candidate champion found: {best_run_name} with CV F1: {mean_cv_f1:.4f}"
            )

        # Lưu Artifact cho run hiện tại
        temp_name = f"temp_{config['name']}.joblib"
        joblib.dump(best_model, temp_name)
        artifact = wandb.Artifact(config["name"], type="model")
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

    champion_path = "models/logistic_model.joblib"
    joblib.dump(best_model_overall, champion_path)

    with wandb.init(
        project="Wine-Quality-Prediction",
        name="logistic_final_champion",
        job_type="archive",
    ) as final_run:
        final_artifact = wandb.Artifact("champion-model", type="model")
        final_artifact.add_file(champion_path)
        final_run.log_artifact(final_artifact)
        final_run.log({"final_best_cv_f1": best_f1_cv})
else:
    print("No healthy models met the minimum F1 threshold.")

# In bảng xếp hạng cuối cùng (Sắp xếp theo CV F1 giảm dần)
print("\n" + "=" * 80)
print("BẢNG XẾP HẠNG ĐỘ ỔN ĐỊNH & CHỈ SỐ THỰC CHIẾN (Sắp xếp theo CV F1)")
print("=" * 80)
df_stability = pd.DataFrame(stability_history)
df_stability = df_stability.sort_values(by="cv_f1", ascending=False)
print(df_stability.to_string(index=False))