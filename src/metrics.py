from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    log_loss  # Thêm cái này để tính Log Loss
)

# Thêm tham số y_proba=None vào định nghĩa hàm
def evaluate_classification(y_test, y_pred, y_proba=None, model_name="Model"):
    print(f"\n===== {model_name} (Classification) =====")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Kiểm tra nếu người dùng có truyền y_proba thì mới tính Log Loss
    if y_proba is not None:
        loss = log_loss(y_test, y_proba)
        print(f"Log Loss:  {loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nNhận xét:")
    if f1 > 0.85:
        print("- Model rất tốt, khả năng phân loại rượu ngon/thường rất chính xác.")
    elif f1 > 0.75:
        print("- Model khá ổn, nhưng có thể cải thiện bằng cách tuning hoặc xử lý dữ liệu.")
    else:
        print("- Model chưa tốt, cần kiểm tra lại feature engineering hoặc bù mẫu (SMOTE).")