import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from Learning_Curve import plot_learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from metrics import evaluate_classification

def load_and_preprocess():
    
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
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    models_to_test = {
        "XGBoost": "models/xgb_model.joblib",
        "Random Forest": "models/rf_model.joblib",
        "SVM": "models/svm_model.joblib",
        "Logistic": "models/logistic_model.joblib"
    }

    # --- CỬA SỔ 1: LEARNING CURVE (2x2) ---
    fig_lc, axes_lc = plt.subplots(2, 2, figsize=(15, 12))
    axes_lc = axes_lc.flatten()

    # --- CỬA SỔ 2: CONFUSION MATRIX (2x2) ---
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
    axes_cm = axes_cm.flatten()

    # --- CỬA SỔ 3: ROC CURVE (CHUNG 1 HÌNH) ---
    fig_roc = plt.figure(figsize=(9, 7)) # Tạo figure mới cho ROC

    for i, (name, path) in enumerate(models_to_test.items()):
        if not os.path.exists(path):
            print(f"⚠️ Cảnh báo: Không tìm thấy {path}")
            continue

        print(f" Đang đánh giá model: {name}...")
        model = joblib.load(path)
        
        # 1. Dự đoán
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 2. In metrics ra terminal
        evaluate_classification(y_test, y_pred, y_proba=model.predict_proba(X_test), model_name=name)

        # 3. Vẽ Confusion Matrix (Vào ô tương ứng của Figure CM)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Thường', 'Ngon'])
        disp.plot(ax=axes_cm[i], cmap=plt.cm.Blues, colorbar=False)
        axes_cm[i].set_title(f"Confusion Matrix: {name}")

        # 4. Vẽ ROC Curve (Vào Figure ROC chung)
        plt.figure(fig_roc.number) # CHỈ ĐỊNH: Nhảy vào đúng figure ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})', linewidth=2)

        # 5. Vẽ Learning Curve (Vào ô tương ứng của Figure LC)
        # Truyền axes_lc[i] vào hàm của bạn
        plot_learning_curve(model, name, X_train, y_train, cv=5, ax=axes_lc[i])

    # --- ĐỊNH DẠNG LẠI CÁC FIGURE SAU VÒNG LẶP ---

    # Hoàn thiện Figure ROC
    plt.figure(fig_roc.number)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5)
    plt.title('So sánh ROC Curves giữa các thuật toán')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Hoàn thiện Figure CM
    fig_cm.tight_layout()
    
    # Hoàn thiện Figure LC
    fig_lc.tight_layout()

    plt.show()