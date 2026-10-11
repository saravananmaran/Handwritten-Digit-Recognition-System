"""
Model evaluation pipeline for Handwritten Digit Recognition System.

Evaluates the trained CNN on the untouched MNIST test dataset.
Generates:
1. Test accuracy and test loss
2. Precision, Recall, and F1-score (macro and weighted)
3. Full classification report (saved to results/classification_report.txt)
4. Confusion matrix visualization (saved to results/confusion_matrix.png)
5. Sample predictions grid with true vs predicted labels (saved to results/sample_predictions.png)
6. Metrics summary JSON (saved to results/evaluation_metrics.json)
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from src.preprocessing import load_and_preprocess_data
from src.model import load_trained_model
from src.device import configure_device



WHY_SEPARATE_TEST_SET = (
    "In rigorous Machine Learning methodology, the test dataset must remain strictly separated and untouched "
    "throughout all training and hyperparameter tuning phases. While the training set optimizes network weights "
    "via gradient descent and the validation set guides architectural choices and early stopping, the test set "
    "serves as an unbiased proxy for real-world unseen data. Exposing test data during training causes data leakage, "
    "leading to over-optimistic metric estimates that fail during real-world inference."
)


def plot_confusion_matrix(cm: np.ndarray, results_dir: str = "results"):
    """Plots and saves an annotated heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = [str(i) for i in range(10)]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix - MNIST CNN Test Set",
        ylabel="Actual True Label",
        xlabel="Model Predicted Label"
    )
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9, fontweight="medium"
            )

    plt.tight_layout()
    save_path = os.path.join(results_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Info] Saved confusion matrix to: {save_path}")


def plot_sample_predictions(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray,
                            results_dir: str = "results", num_samples: int = 15):
    """Plots a grid of test predictions comparing actual vs predicted labels."""
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    axes = axes.flatten()

    indices = np.random.RandomState(42).choice(len(X_test), num_samples, replace=False)

    for idx, sample_i in enumerate(indices):
        ax = axes[idx]
        img = X_test[sample_i].squeeze()
        actual = y_test[sample_i]
        predicted = y_pred[sample_i]

        ax.imshow(img, cmap="gray")
        ax.axis("off")
        color = "#2E7D32" if actual == predicted else "#C62828"
        ax.set_title(f"True: {actual} | Pred: {predicted}", fontsize=11, color=color, fontweight="bold")

    plt.suptitle("Sample Predictions on Untouched Test Set", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "sample_predictions.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Info] Saved sample predictions to: {save_path}")


def evaluate(model_path: str = "models/mnist_cnn.keras", data_dir: str = None, results_dir: str = "results"):
    """
    Evaluates the model on the MNIST test set and writes output artifacts.
    """
    print("=" * 65)
    print(" HANDWRITTEN DIGIT RECOGNITION SYSTEM - MODEL EVALUATION")
    print("=" * 65)

    os.makedirs(results_dir, exist_ok=True)

    # Device configuration (GPU detection, memory growth, CPU fallback)
    device_info = configure_device(verbose=True)
    target_device = device_info["device_string"]
    print(f"\n[Execution Target] Evaluation ops mapped to: {target_device} ({device_info['device_name']})")

    # 1. Load untouched test data
    print("\n[Step 1/4] Loading test dataset...")
    _, _, (X_test, y_test) = load_and_preprocess_data(data_dir=data_dir)
    print(f" -> Test set size: {X_test.shape[0]:,} images, shape: {X_test.shape[1:]}")

    import tensorflow as tf
    with tf.device(target_device):
        # 2. Load model
        print(f"\n[Step 2/4] Loading trained CNN from {model_path}...")
        model = load_trained_model(model_path)

        # 3. Model evaluation on test set
        print(f"\n[Step 3/4] Evaluating test accuracy and loss on {target_device}...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

        y_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)


    # Calculate Scikit-learn metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    clf_report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # 4. Generate and save artifacts
    print("\n[Step 4/4] Generating reports and visual charts...")

    # Save classification report text
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("MNIST CNN MODEL CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(clf_report)
        f.write("\n\n" + "=" * 65 + "\n")
        f.write("WHY TEST SET MUST REMAIN SEPARATE:\n")
        f.write("=" * 65 + "\n")
        f.write(WHY_SEPARATE_TEST_SET + "\n")
    print(f" -> Saved classification report to {report_path}")

    # Save metrics JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_score_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_score_weighted": float(f1_weighted),
        "total_test_samples": int(len(y_test)),
        "correct_predictions": int(np.sum(y_test == y_pred)),
        "misclassifications": int(np.sum(y_test != y_pred)),
        "rationale_test_split": WHY_SEPARATE_TEST_SET
    }
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f" -> Saved metrics JSON to {metrics_path}")

    # Generate charts
    plot_confusion_matrix(cm, results_dir)
    plot_sample_predictions(X_test, y_test, y_pred, results_dir)

    print("\n" + "=" * 65)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f" Test Accuracy  : {test_accuracy * 100:.2f}%")
    print(f" Test Loss      : {test_loss:.4f}")
    print(f" Precision (Macro): {prec_macro * 100:.2f}%")
    print(f" Recall (Macro)   : {rec_macro * 100:.2f}%")
    print(f" F1-Score (Macro) : {f1_macro * 100:.2f}%")
    print("=" * 65)
    print(f"\nAcademic Note on Test Set Separation:\n{WHY_SEPARATE_TEST_SET}")
    print("=" * 65)

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model on MNIST test dataset.")
    parser.add_argument("--model", type=str, default="models/mnist_cnn.keras", help="Path to trained model")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing MNIST dataset")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    evaluate(model_path=args.model, data_dir=args.data_dir, results_dir=args.results_dir)
