import pandas as pd
import wandb
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from metrics import evaluate_classification
from Learning_Curve import plot_learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# 1. LOAD & PREPROCESS DATA
# ==========================================
# Đọc dữ liệu từ file CSV của hai loại rượu đỏ và trắng
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")

# Gán nhãn phân biệt loại rượu (0 cho đỏ, 1 cho trắng) trước khi gộp
df_red["type"] = 0
df_white["type"] = 1 

# Gộp hai tập dữ liệu thành một DataFrame chung
df = pd.concat([df_red, df_white], ignore_index=True)

# Tính toán các đặc trưng hóa học mới dựa trên kiến thức chuyên môn về rượu
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sugar_alcohol_ratio"] = df["residual sugar"] / (df["alcohol"] + 1e-5)
df["density_alcohol_interaction"] = df["density"] * df["alcohol"]

# Chuyển bài toán hồi quy điểm số thành bài toán phân loại nhị phân
# Nhãn 1 (Rượu ngon) nếu điểm >= 7, ngược lại là nhãn 0
df["quality"] = (df["quality"] >= 7).astype(int)

# Chia tách tập dữ liệu thành các đặc trưng (X) và mục tiêu dự báo (y)
X = df.drop("quality", axis=1)
y = df["quality"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80/20
# Stratify đảm bảo tỷ lệ các lớp trong hai tập là tương đương nhau
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================
# 2. EXPERIMENT CONFIGS
# ==========================================
# Danh sách các cấu hình thử nghiệm để so sánh hiệu quả của các phương pháp xử lý dữ liệu và tham số mô hình
configs = [
    {
        "name": "run_1_baseline_basic", 
        "smote_method": "standard", "sampling_ratio": 0.3,
        "param_grid": {
            'rf__n_estimators': [700],
            'rf__max_depth': [10],
            'rf__min_samples_leaf': [40],
            'rf__max_features': ['sqrt']
        }
    },
    {
        "name": "run_2_smote_impact", 
        "smote_method": "standard", "sampling_ratio": 0.35, 
        "param_grid": {
            'rf__n_estimators': [900],
            'rf__max_depth': [8],
            'rf__min_samples_split': [55],
            'rf__max_samples': [0.65]
        }
    },
    {
        "name": "run_3_overfit_control", 
        "smote_method": "borderline", "sampling_ratio": 0.38, 
        "param_grid": {
            'rf__n_estimators': [1000],
            'rf__max_depth': [7],
            'rf__min_samples_split': [60],
            'rf__min_samples_leaf': [35],
            'rf__max_samples': [0.6]
        }
    },
    {
        "name": "run_4_pruning_expert", 
        "smote_method": "borderline", "sampling_ratio": 0.4, 
        "param_grid": {
            'rf__n_estimators': [1100],
            'rf__max_depth': [8],
            'rf__min_samples_split': [50],
            'rf__min_samples_leaf': [25],
            'rf__max_samples': [0.7]
        }
    },
    {
        "name": "run_5_final_anchor", 
        "smote_method": "borderline", "sampling_ratio": 0.45, 
        "param_grid": {
            'rf__n_estimators': [1200],
            'rf__max_depth': [11],
            'rf__min_samples_split': [40],
            'rf__min_samples_leaf': [20],
            'rf__max_samples': [0.78],
            'rf__max_features': ['sqrt']
        }
    }
]

# Các ngưỡng đánh giá để phân loại trạng thái mô hình
MAX_OVERFIT_GAP = 0.08  # Khoảng cách tối đa cho phép giữa F1 train và F1 test
MIN_F1_CV = 0.55         # Ngưỡng F1 tối thiểu để coi là mô hình đạt yêu cầu
best_f1_cv = -1.0
best_model_overall = None
best_run_name = ""
best_run_id = None # Lưu lại ID của run tốt nhất để log artifact sau này

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for run_cfg in configs:
    # Khởi tạo một phiên làm việc mới trên Weights & Biases để theo dõi thí nghiệm
    with wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        name=run_cfg['name'],
        config=run_cfg,
    ) as run:
        
        print(f"Executing: {run_cfg['name']}")

        # Lựa chọn thuật toán tăng cường dữ liệu dựa trên cấu hình (SMOTE hoặc BorderlineSMOTE)
        if run_cfg['smote_method'] == "borderline":
            smote = BorderlineSMOTE(sampling_strategy=run_cfg['sampling_ratio'], random_state=42)
        else:
            smote = SMOTE(sampling_strategy=run_cfg['sampling_ratio'], random_state=42)

        # Xây dựng Pipeline để đảm bảo SMOTE chỉ áp dụng trên tập train trong quá trình Cross-validation
        pipeline = ImbPipeline([
            ('smote', smote),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        
        # Thiết lập chiến lược chia Fold cho Cross-validation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Tìm kiếm bộ tham số tối ưu (Hyperparameter Tuning) bằng GridSearchCV
        grid_search = GridSearchCV(pipeline, run_cfg['param_grid'], cv=kf, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Lưu lại mô hình có tham số tốt nhất từ kết quả tìm kiếm
        best_model = grid_search.best_estimator_
        
        # Cập nhật các tham số tìm được lên hệ thống quản lý thí nghiệm
        run.config.update({"best_params_found": grid_search.best_params_})

        # Thực hiện dự báo trên cả tập huấn luyện và kiểm tra để đánh giá hiệu năng
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)
        y_train_pred = best_model.predict(X_train)

        # Gọi hàm đánh giá chi tiết (Precision, Recall, F1...)
        evaluate_classification(y_test, y_test_pred, y_proba=y_test_proba, model_name=run_cfg['name'])

        # Tính toán các chỉ số F1 và độ chênh lệch để kiểm tra hiện tượng Overfitting
        cv_f1 = grid_search.best_score_
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        overfit_gap = train_f1 - test_f1

        # Gán nhãn trạng thái cho mô hình dựa trên các ngưỡng đã thiết lập
        status_tags = [run_cfg['smote_method'].upper()]
        if overfit_gap > MAX_OVERFIT_GAP: 
            status_tags.append("overfitting")
        elif cv_f1 < MIN_F1_CV: 
            status_tags.append("underfitting")
        else: 
            status_tags.append("healthy")
        run.tags = run.tags + tuple(status_tags)

        # Log Learning Curve
        plot_learning_curve(best_model, run_cfg['name'], X_train, y_train, kf)
        lc_img = wandb.Image(plt)
        plt.close()

        # Trích xuất độ quan trọng của các đặc trưng từ mô hình Random Forest
        importances = best_model.named_steps['rf'].feature_importances_
        indices = np.argsort(importances)[-10:]
        
        # Tạo bảng dữ liệu và biểu đồ cột trên WandB để trực quan hóa tầm quan trọng của đặc trưng
        fi_table = wandb.Table(
            columns=["Feature", "Importance"],
            data=[[X.columns[i], importances[i]] for i in reversed(indices)]
        )

        # Ghi nhận các chỉ số và biểu đồ vào Dashboard của hệ thống quản lý thí nghiệm
        wandb.log({
            "learning_curve": lc_img,
            "best_cv_f1": cv_f1,
            "test_f1": test_f1,
            "train_f1": train_f1,
            "overfit_gap": overfit_gap,
            "smote_ratio": run_cfg['sampling_ratio'],
            "top_10_features_chart": wandb.plot.bar(fi_table, "Feature", "Importance", title="Top 10 Features"),
            "confusion_matrix": wandb.plot.confusion_matrix(y_true=y_test.values, preds=y_test_pred, class_names=["Normal", "Good"]),
            "pr_curve": wandb.plot.pr_curve(y_true=y_test.values, y_probas=y_test_proba, labels=["Normal", "Good"]),
        })

        # Lưu vết mô hình tốt nhất nếu nó thỏa mãn điều kiện "Healthy" và có điểm CV F1 cao nhất
        if "healthy" in status_tags and cv_f1 > best_f1_cv:
            best_f1_cv = cv_f1
            best_model_overall = best_model
            best_run_name = run_cfg['name']
            best_run_id = run.id # Lưu ID lại để dùng cho Artifact sau này
            run.tags = run.tags + ("champion",)
            print(f"New candidate champion found: {best_run_name} with CV F1: {cv_f1:.4f}")

# ==========================================
# 4. EXPORT CHAMPION
# ==========================================
# Nếu tìm thấy mô hình tối ưu nhất sau các lượt chạy, xuất mô hình đó làm phiên bản cuối cùng
if best_model_overall:
    print("-" * 30)
    print(f"FINAL SELECTION: {best_run_name} | Best CV F1: {best_f1_cv:.4f}")
    print("-" * 30)
    
    # Tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_model.joblib"
    metadata_path = "models/rf_metadata.joblib"
    
    metadata_info = {
        "model_name": "Random Forest",
        "metrics":{
            "best_f1_cv": round(best_f1_cv, 4),
        },
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(X.columns) if 'X' in locals() else "unknown"
    }

     # 1. Lưu cục bộ
    joblib.dump(best_model_overall, model_path)
    joblib.dump(metadata_info, metadata_path)
    
    # 2. Lưu lên WandB Artifact
    # Khởi tạo lại run của champion để log artifact hoặc khởi tạo run mới chuyên dụng
    with wandb.init(
        entity="ngphuquy241-tr-ng-i-h-c-m-th-nh-ph-h-ch-minh",
        project="Wine-Quality-Prediction",
        id=best_run_id, 
        resume="must"
    ) as final_run:
        artifact = wandb.Artifact(f"champion_model_{best_run_name}", type='model', metadata=metadata_info)
        artifact.add_file(model_path)
        artifact.add_file(metadata_path)
        final_run.log_artifact(artifact)
        print("Champion model artifact has been logged to WandB.")