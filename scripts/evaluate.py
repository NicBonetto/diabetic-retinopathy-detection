import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent))

from src.core.classifiers import create_model
from src.core.data import DRDataset
from src.utils.metrics import (
    calculate_metrics_with_probs,
    print_metrics_report,
    plot_confusion_matrix
)
from src.utils.visualizations import (
    plot_roc_curves,
    visualize_predictions,
    get_gradcam_viz
)

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list
) -> dict:
    """Evaluate model on dataset."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print('Running evaluation...')
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = calculate_metrics_with_probs(all_labels, all_preds, all_probs)

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'metrics': metrics
    }

def save_results(results: dict, output_dir: Path, class_names: list):
    """Save evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f'✓ Saved metrics to {metrics_file}')

    np.savez(
        output_dir / 'predictions.npz',
        predictions=results['predictions'],
        labels=results['labels'],
        probabilities=results['probabilities']
    )
    print(f'✓ Saved predictions to {output_dir / "predictions.npz"}')

    cm = np.array(results['metrics']['confusion_matrix'])
    fig = plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix (Counts)')
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig = plot_confusion_matrix(cm, class_names, normalize=True, title='Confusion Matrix (Normalized)')
    fig.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved confusion matrices')

    fig = plot_roc_curves(results['labels'], results['probabilities'], class_names)
    fig.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved ROC curves')

def generate_gradcam_samples(
    model: nn.Module,
    dataset: DRDataset,
    results: dict,
    output_dir: Path,
    class_names: list,
    n_samples: int = 5
):
    """Generate Grad-CAM visualizaions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Generating Grad-CAM samples...')

    preds = results['predictions']
    labels = results['labels']
    probs = results['probabilities']

    if hasattr(model.backbone, '__getitem__'):
        target_layer = model.backbone[-1]
    else:
        target_layer = list(model.backbone.children())[-1]

    device = next(model.parameters()).device

    print(' Generating correct predictions...')
    for class_id in range(len(class_names)):
        correct_indices = np.where((labels == class_id) & (preds == class_id))[0]
        if len(correct_indices) == 0:
            continue
        
        confidences = probs[correct_indices, class_id]
        best_idx = correct_indices[np.argmax(confidences)]

        image_tensor, label = dataset[best_idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        from PIL import Image
        image_id = dataset.image_ids[best_idx]
        img_path = dataset.data_dir / f'{image_id}.png'
        original_img = np.array(Image.open(img_path).convert('RGB'))

        gradcam_viz = get_gradcam_viz(
            model=model,
            input_tensor=image_tensor,
            original_image=original_img.astype(np.float32) / 255.0,
            target_layer=target_layer,
            target_class=class_id
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(original_img)
        ax1.set_title(f'Original - {class_names[class_id]}', fontsize=12)
        ax1.axis('off')

        ax2.imshow(gradcam_viz),
        ax2.set_title(f'Grad-CAM (Confidence: {probs[best_idx, class_id]:.2%})')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(
            output_dir / f'gradcam_correct_class{class_id}_{class_names[class_id]}.png'.replace(' ', '_'),
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()

    print(' Generating error cases...')
    incorrect_indices = np.where(preds != labels)[0]
    if len(incorrect_indices) > 0:
        confidences = np.max(probs[incorrect_indices], axis=1)
        worst_errors = incorrect_indices[np.argsort(confidences)[-min(3, len(incorrect_indices)):]]

        for i, idx in enumerate(worst_errors):
            true_class = labels[idx]
            pred_class = preds[idx]
            confidence = probs[idx, pred_class]

            image_tensor, _ = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            image_id = dataset.image_ids[idx]
            img_path = dataset.data_dir / f'{image_id}.png'
            original_img = np.array(Image.open(img_path).convert('RGB'))

            gradcam_viz = get_gradcam_viz(
                model=model,
                input_tensor=image_tensor,
                original_image=original_img.astype(np.float32) / 255.0,
                target_layer=target_layer,
                target_class=pred_class
            )
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(original_img)
            ax1.set_title(f'True: {class_names[true_class]}', fontsize=12, color='green')
            ax1.axis('off')

            ax2.imshow(gradcam_viz)
            ax2.set_title(
                f'Predicted: {class_names[pred_class]} ({confidence:.2%})',
                fontsize=12,
                color='red'
            )
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(
                output_dir / f'gradcam_error_{i + 1}_true{true_class}_pred{pred_class}.png'.replace(' ', '_'),
                dpi=150,
                bbox_inches='tight'
            )
            plt.close()

    print(f'✓ Saved Grad-CAM visualizations to {output_dir}')

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DR model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels CSV')
    parser.add_argument('--output-dir', type=str, default='results/evaluation', help='Output directory')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Model backbone (resnet50, resnet101, efficientnet_b0, efficientnet_b3)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--gradcam-samples', type=int, default=5, help='Number of Grad-CAM samples per class')

    args = parser.parse_args()

    print('=' * 70)
    print('DIABETIC RETINOPATHY MODEL EVALUATION')
    print('=' * 70)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Data directory: {args.data_dir}')
    print(f'Labels file: {args.labels}')
    print(f'Output directory: {args.output_dir}')
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    if 'efficientnet_b0' in args.backbone:
        image_size = 224
    elif 'efficientnet_b3' in args.backbone:
        image_size = 300
    else:
        image_size = 512

    print('Loading model...')
    model = create_model(backbone=args.backbone, num_classes=5, pretrained=False)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print('✓ Model loaded\n')

    print('Loading dataset...')
    dataset = DRDataset(
        data_dir=args.data_dir,
        labels_file=args.labels,
        transform=DRDataset.get_val_transforms(),
        image_size=(image_size, image_size)
    )

    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f'✓ Loaded {len(dataset)} images\n')

    results = evaluate_model(model, data_loader, device, class_names)

    print('\n' + '=' * 70)
    print_metrics_report(results['metrics'], class_names)
    print('=' * 70 + '\n')

    output_dir = Path(args.output_dir)
    save_results(results, output_dir, class_names)

    gradcam_dir = output_dir / 'gradcam_samples'
    generate_gradcam_samples(
        model, dataset, results, gradcam_dir,
        class_names, n_samples=args.gradcam_samples
    )

    print('\n' + '=' * 70)
    print('EVALUATION COMPLETE!')
    print('=' * 70)
    print(f'\nResults saved to: {output_dir}')
    print(' -  metrics.json')
    print(' -  predictions.npz')
    print('  -  confusion_matrix.png')
    print('  -  confusion_matrix_normalized.png')
    print('  -  roc_curves.png')
    print('  -  gradcam_samples/')
    print()

if __name__ == '__main__':
    main()
