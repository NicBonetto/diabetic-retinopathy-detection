import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_auc_score
)
from typing import Dict, Any

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate metrics for DR classification."""
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    sensitivities = []
    specificities = []

    n_classes = cm.shape[0]

    for i in range(n_classes):
        tp = cm[i][i]
        fn = cm[i][:].sum() - tp
        fp = cm[:][i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'kappa': kappa,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'sensitivity': sensitivities,
            'specificity': specificities,
            'support': support.tolist()
        },
        'confusion_matrix': cm.tolist()
    }

def calculate_metrics_with_probs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Any]:
    """Calculate metrics for DR classification including probability scores."""
    metrics = calculate_metrics(y_true, y_pred)

    try:
        auc_scores = []
        n_classes = y_probs.shape[1]

        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                auc = roc_auc_score(y_true_binary, y_probs[:, i])
                auc_scores.append(auc)
            else:
                auc_scores.append(np.nan)

        metrics['per_class']['auc'] = auc_scores
        metrics['auc_macro'] = np.nanmean(auc_scores)
    except Exception as e:
        print(f'Warning - Could not calculate AUC scores: {e}')

    return metrics

def print_metrics_report(metrics: Dict[str, Any], class_names: list=None):
    """Print formatted metrics report."""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['per_class']['precision']))]

    print('\n' + '=' * 60)
    print('CLASSIFICATION METRICS REPORT')
    print('=' * 60)

    print('\nOverall Metrics:')
    print(f'    Accuracy:           {metrics["accuracy"]:4f}')
    print(f'    Macro Precision:    {metrics["precision_macro"]:.4f}')
    print(f'    Macro Recall:       {metrics["recall_macro"]:.4f}')
    print(f'    Macro F1:           {metrics["f1_macro"]:.4f}')
    print(f'    Weighted F1:        {metrics["f1_weighted"]:.4f}')
    print(f'    Cohen\'s Kappa:     {metrics["kappa"]:.4f}')

    if 'auc_macro' in metrics:
        print(f'    Macro AUC:          {metrics["auc_macro"]:.4f}')

    print(f'\nPer-Class Metrics:')
    print(f'{"Class":<20} {"Precision":<12} {"Recall":<12} {"F1":<12} {"Support":<10}')
    print('-' * 60)

    per_class = metrics['per_class']
    for i, name in enumerate(class_names):
        print(
            f'{name:<20} {per_class["precision"][i]:<12.4f}'
            f'{per_class["recall"][i]:<12.4f} {per_class["f1"][i]:<12.4f}'
            f'{int(per_class["support"][i]):<10}'
        )

    print(f'\nClinical Metrics (Sensitivity/Specificity)')
    print(f'{"Class":<20} {"Sensitivity":<15} {"Specificity":<15}')
    print('-' * 50)

    for i, name in enumerate(class_names):
        print(
            f'{name:<20} {per_class["sensitivity"][i]:<15.4f}'
            f'{per_class["specificity"][i]:<15.4f}'
        )

    if 'auc' in per_class:
        print('\nAUC Scores:')
        for i, name in enumerate(class_names):
            auc = per_class['auc'][i]
            if not np.isnan(auc):
                print(f'    {name}: {auc:.4f}')

    print('\n' + '=' * 60)

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    normalize: bool=False,
    title: str='Confusion Matrix'
):
    """Plot confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    return plt.gcf()

