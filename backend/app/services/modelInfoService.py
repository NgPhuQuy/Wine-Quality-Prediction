from . import model, metadata

def get_model_metadata():
    if model is None:
        return {"status": "error", "message": "Mô hình chưa được tải."}
    
    try:
        actual_model = model.steps[-1][1]
        model_type = actual_model.__class__.__name__
        
        n_features = getattr(actual_model, 'n_features_in_', "N/A")
        if n_features != "N/A":
            n_features = int(n_features)
        
        response = {
            "status": "success",
            "model_info": {
                "type": model_type,
                "n_features_in": n_features
            },
            "metadata": metadata if metadata else {}
        }
        
        return response

    except Exception as e:
        return {"status": "error", "message": f"Lỗi đọc thông tin: {str(e)}"}