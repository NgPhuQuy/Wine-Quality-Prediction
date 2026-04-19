from . import model

def get_model_metadata():
    if model is None:
        return {"status": "error", "message": "Mô hình chưa được tải."}
    
    try:
        actual_model = model.steps[-1][1]
        model_name = actual_model.__class__.__name__

        # Lấy n_features_in_
        n_features = getattr(actual_model, 'n_features_in_', "N/A")
        if n_features != "N/A":
            n_features = int(n_features)
            
        return {
            "status": "success",
            "model_name": model_name,
            "n_features_in": n_features
        }
    except Exception as e:
        return {"status": "error", "message": f"Lỗi đọc thông tin: {str(e)}"}