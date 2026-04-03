from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    log_loss
)

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

    if y_proba is not None:
        loss = log_loss(y_test, y_proba)
        print(f"Log Loss:  {loss:.4f}")

    print("\nClassification Report:")
  
    print(classification_report(y_test, y_pred))

    recall_class_1 = report['1']['recall']
    print("Nhận xét")
    if acc > 0.8:
        print(f"- Model {model_name} có độ chính xác rất cao ({acc*100:.1f}%).")
    elif acc >= 0.65:
        print(f"- Model {model_name} có độ chính xác ở mức khá ({acc*100:.1f}%), tạm chấp nhận được.")
    else:
        print(f"- Model {model_name} có độ chính xác thấp ({acc*100:.1f}%), cần tuning mạnh hoặc đổi thuật toán.")
    
    if recall_class_1 > 0.75:
        print(f"- Khả năng nhận diện rượu NGON (Recall lớp 1) cực tốt ({recall_class_1*100:.1f}%).")
    else:
        print(f"- Model còn bỏ sót nhiều rượu ngon, cần chú ý Recall lớp 1.")
    print("-" * 30)