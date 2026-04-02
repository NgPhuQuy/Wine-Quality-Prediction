from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import wandb

def plot_learning_curve(model, model_name, X, y, cv):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Train")
    plt.plot(train_sizes, val_mean, label="Validation")
    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve - {model_name}")
    plt.legend()
    plt.grid()

    wandb.log({f"{model_name}_learning_curve": wandb.Image(plt)})
    plt.show()