import os
import joblib
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
# 1. Load Data
# ==========================================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

df_red["type"] = 0
df_white["type"] = 1

df = pd.concat([df_red, df_white], ignore_index=True)

# ==========================================
# 2. Feature Engineering
# ==========================================
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]

# Gán nhãn nhị phân: Chất lượng >= 7 là 1, ngược lại là 0
df["quality"] = (df["quality"] >= 7).astype(int)

X = df.drop("quality", axis=1)
y = df["quality"]

# ==========================================
# 3. Split Data
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ==========================================
# FUNCTION METRIC
# ==========================================
def compute_metric(metric, y_true, y_pred):
    if metric == "f1":
        return f1_score(y_true, y_pred)
    elif metric == "precision":
        return precision_score(y_true, y_pred)
    elif metric == "recall":
        return recall_score(y_true, y_pred)


def custom_score_func(y_true, y_proba, metric_name, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return compute_metric(metric_name, y_true, y_pred)


def custom_scorer(metric_name, threshold):
    return make_scorer(
        custom_score_func,
        response_method="predict_proba",
        metric_name=metric_name,
        threshold=threshold,
    )


# ==========================================
# 4. Config 7 RUNS ĐẦY ĐỦ
# ==========================================
configs = [
    {
        "name": "logistic_run3",
        "metric": "recall",
        "class_weight": None,
        "smote": False,
        "threshold": 0.2,
        "penalty": "l2",
        "desc": "GIỮ LẠI RUN 3: Hạ mạnh ngưỡng xuống 0.2 để bùng nổ Recall.",
    },
    {
        "name": "logistic_run5",
        "metric": "recall",
        "class_weight": {0: 1, 1: 3},
        "smote": False,
        "threshold": 0.5,
        "penalty": "l2",
        "desc": "GIỮ LẠI RUN 5: Phạt cân bằng vừa đủ 1:3 với ngưỡng chuẩn 0.5.",
    },
    {
        "name": "logistic_run1",
        "metric": "recall",
        "class_weight": None,
        "smote": False,
        "threshold": 0.3,
        "penalty": "l2",
        "desc": "Vùng đệm 0.3: Tìm điểm cân bằng tốt hơn mức 0.2 của Run 3.",
    },
    {
        "name": "logistic_run2",
        "metric": "recall",
        "class_weight": None,
        "smote": False,
        "threshold": 0.25,
        "penalty": "l2",
        "desc": "Nhích nhẹ lên 0.25: Xem F1-score có cải thiện so với mức 0.2 không.",
    },
    {
        "name": "logistic_run4",
        "metric": "recall",
        "class_weight": {0: 1, 1: 2.5},
        "smote": False,
        "threshold": 0.5,
        "penalty": "l2",
        "desc": "Nấc đệm 1:2.5: Giảm nhẹ độ phạt so với Run 5 để kéo lại Precision.",
    },
    {
        "name": "logistic_run6",
        "metric": "recall",
        "class_weight": {0: 1, 1: 3.5},
        "smote": False,
        "threshold": 0.5,
        "penalty": "l2",
        "desc": "Nấc đệm 1:3.5: Tăng nhẹ độ phạt so với Run 5 để đẩy thêm Recall.",
    },
    {
        "name": "logistic_run7",
        "metric": "recall",
        "class_weight": {0: 1, 1: 2.5},
        "smote": False,
        "threshold": 0.3,
        "penalty": "l2",
        "desc": "Sự kết hợp hoàn hảo: Phạt nhẹ 1:2.5 và hạ ngưỡng vừa vặn xuống 0.3.",
    },
]

os.makedirs("models", exist_ok=True)
stability_history = []

# ==========================================
# 5. LOOP RUNS
# ==========================================
for config in configs:
    wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        name=config["name"],
        group="logistic_7_runs",
        tags=["logistic", config["metric"]],
    )
    wandb.config.update(config)

    print(f"\n===== RUN: {config['name']} =====")
    print(f"Mục tiêu: {config['desc']}")

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
        lr_model = LogisticRegression(**lr_params)
    else:
        lr_params["solver"] = "lbfgs"
        lr_params["l1_ratio"] = 0
        lr_model = LogisticRegression(**lr_params)

    steps.append(("lr", lr_model))
    pipeline = ImbPipeline(steps)

    # ------------------------------------------
    # Cross Validation (Tracking AUC)
    # ------------------------------------------
    print("--- Cross Validation (Tracking AUC) ---")
    cv_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_val_proba)
        cv_auc_scores.append(auc_score)
        print(f"Fold {fold} AUC: {auc_score:.4f}")

    mean_cv_auc = np.mean(cv_auc_scores)
    std_cv_auc = np.std(cv_auc_scores)
    print(f">> Mean CV AUC: {mean_cv_auc:.4f} (±{std_cv_auc:.4f})")

    # ------------------------------------------
    # GridSearch
    # ------------------------------------------
    param_grid = {"lr__C": [0.01, 0.1, 1, 10]}

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=kf,
        scoring=custom_scorer(config["metric"], config["threshold"]),
        n_jobs=-1,
    )

    print("--- GridSearch ---")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # ------------------------------------------
    # Test Evaluation & Tính toán thông số
    # ------------------------------------------
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= config["threshold"]).astype(int)

    test_metric_score = compute_metric(config["metric"], y_test, y_pred)
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
    # Log WandB
    # ------------------------------------------
    wandb.log(
        {
            "mean_cv_auc": mean_cv_auc,
            "cv_std_auc": std_cv_auc,
            f"test_{config['metric']}": test_metric_score,
            "test_auc": test_auc,
            "overfit_gap": overfit_gap,
            "test_f1": t_f1,
            "test_precision": t_prec,
            "test_recall": t_rec,
            "test_accuracy": t_acc,
        }
    )

    # Lưu model (Có ghi đè nén)
    joblib.dump(best_model, f"models/{config['name']}.joblib", compress=3)

    # Learning Curve
    plot_learning_curve(best_model, config["name"], X_train, y_train, kf)
    wandb.finish()

# ==========================================
# FINAL MODEL SELECTION (BẢNG XẾP HẠNG 9 CỘT)
# ==========================================
print("\n" + "=" * 80)
print("BẢNG XẾP HẠNG ĐỘỔN ĐỊNH & CHỈ SỐ THỰC CHIẾN (Sắp xếp theo Overfit Gap)")
print("=" * 80)

df_stability = pd.DataFrame(stability_history)
df_stability = df_stability.sort_values(by="overfit_gap")

# In ra toàn bộ các cột đầy đủ
print(df_stability.to_string(index=False))

best_stable_run = df_stability.iloc[0]["name"]
print("\n" + "=" * 80)
print(f"🏆 MODEL ỔN ĐỊNH NHẤT: {best_stable_run}")
print("Ý nghĩa: Có khoảng cách giữa CV và Test nhỏ nhất, chống học vẹt!")
print("=" * 80)