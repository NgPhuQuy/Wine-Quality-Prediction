import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import matplotlib
matplotlib.use('Agg') 
from datetime import datetime
from Learning_Curve import plot_learning_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from metrics import evaluate_classification
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD & PREPROCESS DATA
# ==========================================
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")
df_red["type"], df_white["type"] = 0, 1
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
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 2. CONFIGURATIONS
# ==========================================
configs = [
    # --- NHÓM 1: PHÁT TRIỂN TỪ RUN 7 (Ưu tiên F1-Score & Cân bằng) ---
    {
        "name": "svm_v2_run1", 
        "kernel": "rbf", "C": 100.0, "smote": True, "threshold": 0.65 # Hạ nhẹ threshold của run 7 để tăng Recall
    },
    {
        "name": "svm_v2_run2", 
        "kernel": "rbf", "C": 150.0, "smote": True, "threshold": 0.7  # Tăng C để model học quyết liệt hơn
    },
    {
        "name": "svm_v2_run3", 
        "kernel": "rbf", "C": 100.0, "smote": True, "threshold": 0.7  # GIỮ NGUYÊN RUN 7 GỐC
    },

    # --- NHÓM 2: PHÁT TRIỂN TỪ RUN 3 & 5 (Ưu tiên Recall - Bắt rượu tốt) ---
    {
        "name": "svm_v2_run4", 
        "kernel": "rbf", "C": 1.0, "smote": True, "threshold": 0.5   # Cải tiến Run 3: Hạ threshold để Recall > 0.7
    }
    
]

MAX_OVERFIT_GAP = 0.1
MIN_F1_CV = 0.5
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
        tags=["svm", config["kernel"], "smote" if config["smote"] else "no-smote"],
    ) as run:

        print(f"\n🚀 Executing SVM: {config['name']}")

        # Setup Pipeline
        steps = [("scaler", StandardScaler())]
        if config["smote"]:
            steps.append(("smote", SMOTE(random_state=42)))
        
        steps.append(("svm", SVC(
            kernel=config["kernel"], 
            C=config["C"], 
            probability=True, 
            class_weight="balanced", 
            cache_size=1000,
            random_state=42
        )))
        pipeline = ImbPipeline(steps)

        # Cross Validation Manual (AUC & F1)
        cv_f1_scores, cv_auc_scores = [], []
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_val_proba = pipeline.decision_function(X_val)
            
            cv_auc_scores.append(roc_auc_score(y_val, y_val_proba))
            cv_f1_scores.append(f1_score(y_val, (y_val_proba >= config["threshold"]).astype(int)))

        mean_cv_f1 = np.mean(cv_f1_scores)
        mean_cv_auc = np.mean(cv_auc_scores)
        std_cv_auc = np.std(cv_auc_scores)

        # Huấn luyện trên toàn tập train
        pipeline.fit(X_train, y_train)
        
        # Test Evaluation
        y_proba = pipeline.decision_function(X_test)
        y_pred = (y_proba >= config["threshold"]).astype(int)

        t_f1 = f1_score(y_test, y_pred)
        t_auc = roc_auc_score(y_test, y_proba)
        overfit_gap = abs(mean_cv_auc - t_auc)

        # Log History
        stability_history.append({
            "name": config["name"],
            "cv_f1": round(mean_cv_f1, 4),
            "test_f1": round(t_f1, 4),
            "overfit_gap": round(overfit_gap, 4),
            "prec": round(precision_score(y_test, y_pred), 4),
            "rec": round(recall_score(y_test, y_pred), 4),
            "acc": round(accuracy_score(y_test, y_pred), 4),
            "cv_std_auc": round(std_cv_auc, 4)
        })

        # --- Gắn Tag Trạng Thái ---
        status_tags = []
        if overfit_gap > MAX_OVERFIT_GAP: status_tags.append("overfitting")
        if mean_cv_f1 < MIN_F1_CV: status_tags.append("underfitting")
        if not status_tags: status_tags.append("healthy")
        run.tags = run.tags + tuple(status_tags)

        # 1. Log Learning Curve
        plot_learning_curve(pipeline, config["name"], X_train, y_train, kf)
        lc_img = wandb.Image(plt.gcf())
        plt.close()

        # 2. Permutation Importance
        r = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)
        indices = np.argsort(r.importances_mean)[-10:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), r.importances_mean[indices], color="lightgreen")
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.title(f"Permutation Importance - {config['name']}")
        fi_img = wandb.Image(plt.gcf())
        plt.close()

        # 3. Log WandB
        wandb.log({
            "learning_curve": lc_img,
            "feature_importance": fi_img,
            "mean_cv_f1": mean_cv_f1,
            "mean_cv_auc": mean_cv_auc,
            "overfit_gap": overfit_gap,
            "test_f1": t_f1,
            "test_auc": t_auc,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_test.values, preds=y_pred, class_names=["Normal", "Good"]
            ),
            "pr_curve": wandb.plot.pr_curve(
                y_true=y_test.values, 
                y_probas=pipeline.predict_proba(X_test),
                labels=["Normal", "Good"]
            ),
        })

        # Cập nhật Champion (Ưu tiên Healthy + F1 tốt nhất)
        if "healthy" in status_tags and mean_cv_f1 > best_f1_cv:
            best_f1_cv = mean_cv_f1
            best_model_overall = pipeline
            best_run_name = config["name"]
            run.tags = run.tags + ("champion",)
            print(f"⭐ New candidate champion: {best_run_name} (F1: {mean_cv_f1:.4f})")

        # Lưu Artifact từng run
        temp_path = f"temp_{config['name']}.joblib"
        joblib.dump(pipeline, temp_path)
        artifact = wandb.Artifact(config["name"], type="model")
        artifact.add_file(temp_path)
        run.log_artifact(artifact)
        os.remove(temp_path)

# ==========================================
# 4. FINAL CHAMPION & REPORT
# ==========================================
if best_model_overall:
    print("-" * 30)
    print(f"FINAL SELECTION: {best_run_name} | Best CV F1: {best_f1_cv:.4f}")
    print("-" * 30)

    # 1. Cấu hình đường dẫn cho SVM
    os.makedirs("models", exist_ok=True)
    pipeline_path = "models/svm_model.joblib" 
    metadata_path = "metadata/svm_metadata.joblib"

    # 2. Metadata chi tiết (Đúng yêu cầu: metrics, date, version)
    metadata_info = {
        "model_type": "Support Vector Machine (SVM)",
        "version": "1.0.0",
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "best_cv_f1": round(best_f1_cv, 4),
            "status": "healthy"
        },
        "best_run_name": best_run_name,
        "features": X.columns.tolist()
    }

    # 3. Lưu cục bộ Full Pipeline (SMOTE + SVM) và Metadata bằng joblib
    joblib.dump(best_model_overall, pipeline_path)
    joblib.dump(metadata_info, metadata_path)
    print(f"Saved SVM Full Pipeline and Metadata to {pipeline_path}")

    # 4. Lưu lên WandB Artifact
    # Lưu ý: id=best_run_id giúp lưu model vào đúng Run tốt nhất bạn đã thực hiện
    with wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        resume="allow"
    ) as final_run:
        
        artifact = wandb.Artifact(
            name=f"champion_pipeline_{best_run_name.replace(' ', '_')}", 
            type="model_pipeline", 
            metadata=metadata_info
        )
        
        artifact.add_file(pipeline_path)
        artifact.add_file(metadata_path)
        
        final_run.log_artifact(artifact)
        final_run.log({"final_best_cv_f1": best_f1_cv})
        print("SVM Champion artifact has been logged to WandB.")
else:
    print("No healthy models met the minimum F1 threshold.")

# ==========================================
# 5. BẢNG XẾP HẠNG
# ==========================================
print("\n" + "=" * 105)
print(f"{'BẢNG XẾP HẠNG SVM (ƯU TIÊN ĐỘ ỔN ĐỊNH - GAP THẤP)':^105}")
print("=" * 105)

# Sắp xếp theo Gap (thấp đến cao) rồi đến CV F1 (cao đến thấp)
if not stability_history:
    print("No history recorded.")
else:
    df_stability = pd.DataFrame(stability_history).sort_values(
        by=["overfit_gap", "cv_f1"], 
        ascending=[True, False]
    )
    print(df_stability.to_string(index=False))
print("=" * 105)