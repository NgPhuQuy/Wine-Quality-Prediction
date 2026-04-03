import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from metrics import evaluate_classification

def load_and_preprocess():
    if not os.path.exists("data/winequality-red.csv"):
        print(" Lỗi: Không tìm thấy file dữ liệu trong thư mục data/")
        return None, None

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

    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, y_test


if __name__ == "__main__":
    X_test, y_test = load_and_preprocess()
    
    if X_test is not None:
       
        models_to_test = {
            "XGBoost": "models/xgb_model.joblib",
            "Random Forest": "models/rf_model.joblib",
            "SVM": "models/svm_model.joblib",
            "Logistic": "models/logistic_model.joblib"
        }

       
        fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
        axes_cm = axes_cm.flatten()
        
  
        plt.figure(2, figsize=(9, 7))

        for i, (name, path) in enumerate(models_to_test.items()):
            if os.path.exists(path):
                print(f" Đang đánh giá model: {name}...")
                model = joblib.load(path)
                
               
                y_pred = model.predict(X_test)
               
                y_proba = model.predict_proba(X_test)[:, 1]

              
                evaluate_classification(y_test, y_pred, y_proba=model.predict_proba(X_test), model_name=name)

                
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Thường', 'Ngon'])
                disp.plot(ax=axes_cm[i], cmap=plt.cm.Blues, colorbar=False)
                axes_cm[i].set_title(f"Confusion Matrix: {name}")

                
                plt.figure(2)
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', linewidth=2)
            else:
                print(f"⚠️ Cảnh báo: Không tìm thấy file {path}")

      
        plt.figure(2)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Tỉ lệ báo nhầm)')
        plt.ylabel('True Positive Rate (Tỉ lệ bắt đúng)')
        plt.title('So sánh ROC Curves giữa các thuật toán (Tuần 5)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

   
        fig_cm.tight_layout()
        plt.show()
        
        print("\n Hoàn thành demo.")