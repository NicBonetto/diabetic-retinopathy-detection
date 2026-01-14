import torch
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_gradcam_viz(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_layer: torch.nn.Module,
    target_class: Optional[int] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """Generate Grad-CAM visualization."""
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor, targets=None if target_class is None else [target_class])
    grayscale_cam = grayscale_cam[0, :]

    viz = show_cam_on_image(original_image, grayscale_cam, use_rgb=True, image_weight=alpha)
    return viz

def plot_training_history(history: dict, save_path: Optional[str]=None):
    """Plot training and validation metrics over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def visualize_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    class_names: list,
    n_samples: int = 16
):
    """Visualize sample predictions with confidence scores."""
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for idx in range(n_samples):
        ax = axes[idx]
        ax.imshow(images[idx])

        true_label = class_names[true_labels[idx]]
        pred_label = class_names[pred_labels[idx]]
        confidence = pred_probs[idx, pred_labels[idx]]

        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        title = f'True: {true_label}\n Pred: {pred_label}\nConf: {confidence:.2%}'
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig

def plot_class_distribution(labels: np.ndarray, class_names: list, title: str='Class Distribution'):
    """Plot distribution of classes in dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    unique, counts = np.unique(labels, return_counts=True)

    bars = ax.bar(range(len(unique)), counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{int(count)}',
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    return fig

def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, class_names: list):
    """Plot ROC curves for each class."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)

    y_true_binary = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - One vs Rest', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

