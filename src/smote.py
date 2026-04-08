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
from sklearn.metrics import f1_score

# QUAN TRỌNG: Sử dụng Pipeline và SMOTE từ imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# 1. LOAD & PREPROCESS DATA
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
# 2. CONFIGURATIONS FOR SMOTE RUNS
# ==========================================
# Thử nghiệm các mức độ SMOTE khác nhau (sampling_strategy)
# ==========================================
# 2. OPTIMIZED CONFIGURATIONS (Anti-Overfitting)
# ==========================================
# ==========================================
# 2. DIVERSE OPTIMIZED CONFIGS
# ==========================================
configs = [
    # HƯỚNG 1: "THE SPECIALIST" - Ưu tiên Precision cực cao
    # Phù hợp nếu bạn muốn mô hình chỉ "phán" khi cực kỳ chắc chắn là rượu ngon
    {
        "name": "rf_smote_precision_focus", 
        "sampling_ratio": 0.35, # Ít dữ liệu ảo hơn để giữ nguyên bản chất data
        "params": {
            'rf__n_estimators': [1500],
            'rf__max_depth': [9],
            'rf__min_samples_split': [70], # Split rất khắt khe
            'rf__min_samples_leaf': [35],  # Lá lớn để tránh nhiễu
            'rf__max_features': [0.15],    # Mỗi split chỉ xem xét rất ít tính năng
            'rf__max_samples': [0.65]       # Tính ngẫu nhiên cực cao
        }
    },
    
    # HƯỚNG 2: "THE EXPLORER" - Thử nghiệm k-Neighbors của SMOTE
    # Chỉnh k_neighbors trong SMOTE (cần sửa code khởi tạo Pipeline một chút)
    {
        "name": "rf_smote_k_neighbors_adj", 
        "sampling_ratio": 0.45,
        "params": {
            'rf__n_estimators': [1000],
            'rf__max_depth': [10],
            'rf__min_samples_split': [40],
            'rf__min_samples_leaf': [20],
            'rf__max_features': ['log2'],  # Thử log2 thay vì sqrt
            'rf__max_samples': [0.8]
        }
    },

    # HƯỚNG 3: "THE STABILIZER" - Cân bằng giữa các Run V3 bạn đã chạy
    {
        "name": "rf_smote_v4_ultra_stable", 
        "sampling_ratio": 0.4,
        "params": {
            'rf__n_estimators': [2000],    # Tăng mạnh số lượng cây để làm mượt dự đoán
            'rf__max_depth': [7],          # Cây rất nông để chống overfit tuyệt đối
            'rf__min_samples_split': [50],
            'rf__min_samples_leaf': [25],
            'rf__max_features': [0.3],     # Cho phép mỗi cây nhìn rộng hơn một chút
            'rf__max_samples': [0.7]
        }
    }
]

MAX_OVERFIT_GAP = 0.1 
MIN_F1_CV = 0.60 
best_f1_cv = -1.0
best_model_overall = None
best_run_name = ""

# ==========================================
# 3. TRAINING LOOP WITH SMOTE
# ==========================================
for run_cfg in configs:
    with wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        name=run_cfg['name'],
        config=run_cfg
    ) as run:
        
        print(f"Executing SMOTE Run: {run_cfg['name']}")

        # Pipeline tích hợp SMOTE
        # sampling_strategy: tỉ lệ giữa lớp thiểu số/đa số sau khi resample
        # pipeline = ImbPipeline([
        #     ('smote', SMOTE(sampling_strategy=run_cfg['sampling_ratio'], random_state=42)),
        #     ('rf', RandomForestClassifier(random_state=42)) 
        # ])
        # Thêm k_neighbors=3 thay vì mặc định là 5 để SMOTE tập trung vào các nhóm nhỏ hơn
        pipeline = ImbPipeline([
            ('smote', SMOTE(sampling_strategy=run_cfg['sampling_ratio'], k_neighbors=3, random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)) 
        ])
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(pipeline, run_cfg['params'], cv=kf, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Đánh giá
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)
        y_train_pred = best_model.predict(X_train)

        evaluate_classification(y_test, y_test_pred, y_proba=y_test_proba, model_name=run_cfg['name'])

        cv_f1 = grid_search.best_score_
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        overfit_gap = train_f1 - test_f1

        # Gán Tags
        status_tags = ["SMOTE_Method"]
        if overfit_gap > MAX_OVERFIT_GAP: status_tags.append("overfitting")
        if cv_f1 < MIN_F1_CV: status_tags.append("underfitting")
        if len(status_tags) == 1: status_tags.append("healthy")
        run.tags = run.tags + tuple(status_tags)

        # 1. Log Learning Curve
        plot_learning_curve(best_model, run_cfg['name'], X_train, y_train, kf)
        lc_img = wandb.Image(plt)
        plt.close()

        # 2. Xử lý Feature Importance
        importances = best_model.named_steps['rf'].feature_importances_
        features = X.columns
        indices = np.argsort(importances)[-10:]
        
        fi_table = wandb.Table(
            columns=["Feature", "Importance"],
            data=[[features[i], importances[i]] for i in reversed(indices)]
        )

        # 3. Tổng hợp log lên WandB
        wandb.log({
            "learning_curve": lc_img,
            "top_10_features_chart": wandb.plot.bar(fi_table, "Feature", "Importance", title="Top 10 Features (SMOTE)"),
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_test.values, preds=y_test_pred, class_names=["Normal", "Good"]),
            "pr_curve": wandb.plot.pr_curve(y_true=y_test.values, y_probas=y_test_proba, labels=["Normal", "Good"]),
            "best_cv_f1": cv_f1,
            "overfit_gap": overfit_gap,
            "test_f1": test_f1,
            "train_f1": train_f1,
            "smote_ratio": run_cfg['sampling_ratio']
        })

        if "healthy" in status_tags and cv_f1 > best_f1_cv:
            best_f1_cv = cv_f1
            best_model_overall = best_model
            best_run_name = run_cfg['name']
            run.tags = run.tags + ("champion",)

        # Lưu Artifact
        temp_name = f"temp_smote_{run_cfg['name']}.joblib"
        joblib.dump(best_model, temp_name)
        artifact = wandb.Artifact(run_cfg['name'], type='model')
        artifact.add_file(temp_name)
        wandb.log_artifact(artifact)
        os.remove(temp_name)

# ==========================================
# 4. EXPORT CHAMPION
# ==========================================
if best_model_overall:
    print(f"\n--- BEST SMOTE MODEL: {best_run_name} (CV F1: {best_f1_cv:.4f}) ---")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_overall, "models/rf_smote_best_model.joblib")