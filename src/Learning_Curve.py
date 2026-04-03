from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import wandb


def plot_learning_curve(model, model_name, X, y, cv, ax=None): # Để ax=None làm mặc định
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # KIỂM TRA: Nếu ax là None thì tự tạo Figure và Axes mới
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_standalone = True # Đánh dấu đây là vẽ đơn lẻ
    else:
        is_standalone = False # Vẽ gộp vào subplots có sẵn

    ax.plot(train_sizes, train_mean, 'o-', label="Train")
    ax.plot(train_sizes, val_mean, 's-', label="Validation")
    ax.set_xlabel("Training size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve - {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Chỉ Show và Log WandB nếu vẽ đơn lẻ
    if is_standalone:
        if wandb.run is not None:
            # Lưu ảnh vào WandB
            wandb.log({f"{model_name}_learning_curve": wandb.Image(plt)})
        plt.show()