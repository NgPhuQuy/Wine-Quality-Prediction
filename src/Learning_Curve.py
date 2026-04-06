from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import wandb


def plot_learning_curve(model, model_name, X, y, cv, ax=None):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # Nếu không truyền ax, tự tạo một Figure mới để log riêng lẻ
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True

    ax.plot(train_sizes, train_mean, 'o-', label="Train")
    ax.plot(train_sizes, val_mean, 's-', label="Validation")
    ax.set_xlabel("Training size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve - {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Log lên WandB nếu đang trong một Run
    if wandb.run is not None:
        # Nếu là standalone, log trực tiếp figure hiện tại
        # Nếu là một phần của subplot (vẽ 4 model), việc log sẽ do code bên ngoài đảm nhận sau khi vẽ xong cả 4
        if standalone:
            wandb.log({f"learning_curve_{model_name}": wandb.Image(fig)})
            plt.close(fig) # Đóng để giải phóng bộ nhớ

    return ax # Trả về ax để có thể dùng cho việc vẽ gộp 2x2 bên ngoài