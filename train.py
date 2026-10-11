"""
Training pipeline for Handwritten Digit Recognition System Using CNN.

Performs:
1. Loading MNIST dataset (from local ubyte files or Keras fallback).
2. Data normalization (0-255 -> 0-1) and reshaping to (N, 28, 28, 1).
3. Train/Validation split (50,000 train / 10,000 val).
4. CNN architecture construction and compilation.
5. Model training with validation and EarlyStopping.
6. Saving trained model to models/mnist_cnn.keras.
7. Plotting and saving training curves to results/.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import load_and_preprocess_data
from src.model import build_cnn_model, compile_model
from src.device import configure_device



def plot_and_save_training_curves(history_dict: dict, results_dir: str = "results"):
    """Generates and saves professional accuracy and loss curves."""
    os.makedirs(results_dir, exist_ok=True)
    epochs = range(1, len(history_dict["accuracy"]) + 1)

    # 1. Training & Validation Accuracy curve
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(epochs, history_dict["accuracy"], "o-", color="#1E88E5", linewidth=2.2, label="Training Accuracy")
    if "val_accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["val_accuracy"], "s--", color="#43A047", linewidth=2.2, label="Validation Accuracy")
    ax_acc.set_title("CNN Model Accuracy per Epoch", fontsize=14, fontweight="bold", pad=12)
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    ax_acc.set_ylim([0.85, 1.01])
    ax_acc.legend(fontsize=11, loc="lower right")
    ax_acc.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    acc_path = os.path.join(results_dir, "training_accuracy.png")
    fig_acc.savefig(acc_path, dpi=300)
    plt.close(fig_acc)
    print(f"[Info] Saved accuracy curve to: {acc_path}")

    # 2. Training & Validation Loss curve
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, history_dict["loss"], "o-", color="#E53935", linewidth=2.2, label="Training Loss")
    if "val_loss" in history_dict:
        ax_loss.plot(epochs, history_dict["val_loss"], "s--", color="#FB8C00", linewidth=2.2, label="Validation Loss")
    ax_loss.set_title("CNN Model Cross-Entropy Loss per Epoch", fontsize=14, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.legend(fontsize=11, loc="upper right")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(results_dir, "training_loss.png")
    fig_loss.savefig(loss_path, dpi=300)
    plt.close(fig_loss)
    print(f"[Info] Saved loss curve to: {loss_path}")


def train(epochs: int = 10, batch_size: int = 128, data_dir: str = None,
          model_dir: str = "models", results_dir: str = "results", random_seed: int = 42):
    """
    Executes end-to-end model training.
    """
    import tensorflow as tf

    # Set seeds for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    print("=" * 65)
    print(" HANDWRITTEN DIGIT RECOGNITION SYSTEM - CNN TRAINING")
    print("=" * 65)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Device configuration (GPU detection, memory growth, CPU fallback)
    device_info = configure_device(verbose=True)
    target_device = device_info["device_string"]
    print(f"\n[Execution Target] Training ops mapped to: {target_device} ({device_info['device_name']})")

    # 1. Load and prepare dataset
    print("\n[Step 1/5] Loading and preprocessing MNIST dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(
        data_dir=data_dir,
        val_split=0.1666667,  # 10,000 validation samples from 60,000 total train
        random_seed=random_seed
    )
    print(f" -> Training set   : {X_train.shape[0]:,} samples, shape: {X_train.shape[1:]}")
    print(f" -> Validation set : {X_val.shape[0]:,} samples, shape: {X_val.shape[1:]}")
    print(f" -> Test set (held-out): {X_test.shape[0]:,} samples, shape: {X_test.shape[1:]}")

    with tf.device(target_device):
        # 2. Build model architecture
        print("\n[Step 2/5] Building CNN architecture...")
        model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.3)
        compile_model(model, learning_rate=0.001)
        model.summary()

        # 3. Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        # 4. Train model
        print(f"\n[Step 3/5] Training CNN for up to {epochs} epochs (batch_size={batch_size}) on {target_device}...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )


    # 5. Save model in native Keras format
    model_save_path = os.path.join(model_dir, "mnist_cnn.keras")
    print(f"\n[Step 4/5] Saving trained model to {model_save_path}...")
    model.save(model_save_path)
    print(f" -> Successfully saved model to {model_save_path}")

    # 6. Save training history and plots
    print("\n[Step 5/5] Saving training history and generating visualization curves...")
    history_serializable = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=4)
    print(f" -> Saved training history to {history_path}")

    plot_and_save_training_curves(history_serializable, results_dir)

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETED SUCCESSFULLY")
    print(f" Best Validation Accuracy: {max(history_serializable.get('val_accuracy', [0.0])) * 100:.2f}%")
    print(" Next step: Run 'python evaluate.py' to test on untouched test data.")
    print("=" * 65)

    return model, history_serializable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model for MNIST handwritten digit recognition.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing MNIST dataset")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save trained model")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation curves")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir
    )
