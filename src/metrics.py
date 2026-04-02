# metrics.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================
# Classification
# =========================
def evaluate_classification(y_test, y_pred, model_name="Model"):
    print(f"\n===== {model_name} (Classification) =====")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Nhận xét (viết giống sinh viên)
    print("\nNhận xét:")
    if acc > 0.85:
        print("- Model rất tốt, dự đoán chính xác cao.")
    elif acc > 0.75:
        print("- Model khá ổn, có thể cải thiện thêm.")
    else:
        print("- Model chưa tốt, cần tuning thêm.")

# =========================
# Regression
# =========================
def evaluate_regression(y_test, y_pred, model_name="Model"):
    print(f"\n===== {model_name} (Regression) =====")

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    print("\nNhận xét:")
    if r2 > 0.8:
        print("- Model hồi quy rất tốt.")
    elif r2 > 0.6:
        print("- Model khá ổn.")
    else:
        print("- Model còn sai số cao.")