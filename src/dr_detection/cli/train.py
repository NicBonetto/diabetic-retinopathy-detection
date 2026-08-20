import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from dr_detection.core.classifiers import create_model
from dr_detection.core.data import (
    DEFAULT_BEN_GRAHAM_SIGMA,
    DEFAULT_TARGET_RADIUS,
    DRDataset,
)
from dr_detection.core.trainer import Trainer
from dr_detection.utils.visualizations import plot_training_history


def set_seed(seed: int) -> None:
    """
    Seed every RNG that affects a training run.

    torch alone is not enough: albumentations draws from numpy and Python's
    random, and cuDNN picks nondeterministic kernels by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Re-seed numpy and random inside each DataLoader worker.

    Workers are forked after seeding, so without this every worker inherits
    the same numpy/random state and augmentations repeat across workers.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def setup_data_loaders(config: dict) -> tuple:
    """Set up training and validation data loaders."""
    # Shared generator so shuffling and weighted sampling are reproducible.
    generator = torch.Generator()
    generator.manual_seed(config.get('seed', 42))

    image_size = tuple(config['data']['image_size'])

    # 'enabled' controls whether the pipeline runs at load time; 'ben_graham'
    # describes whether it is in play at all. When images were baked with
    # `dr-preprocess --preprocess-images` you want enabled=false but
    # ben_graham=true, so the augmentation pipeline still drops colour jitter.
    prep = config['data'].get('preprocess', {})
    ben_graham = prep.get('ben_graham', True)
    prep_kwargs = {
        'preprocess': prep.get('enabled', True),
        'target_radius': prep.get('target_radius', DEFAULT_TARGET_RADIUS),
        'apply_ben_graham': ben_graham,
        'ben_graham_sigma': prep.get('ben_graham_sigma', DEFAULT_BEN_GRAHAM_SIGMA),
    }

    if prep_kwargs['preprocess']:
        print(f'Fundus preprocessing on at load time '
              f'(target_radius={prep_kwargs["target_radius"]}, '
              f'ben_graham={ben_graham})')
    else:
        print('Fundus preprocessing off at load time (expecting baked images)')

    train_dataset = DRDataset(
        data_dir=config['data']['train_dir'],
        labels_file=config['data']['train_labels'],
        transform=DRDataset.get_train_transforms(
            image_size=image_size,
            ben_graham=ben_graham
        ),
        image_size=image_size,
        **prep_kwargs
    )

    val_dataset = DRDataset(
        data_dir=config['data']['val_dir'],
        labels_file=config['data']['val_labels'],
        transform=DRDataset.get_val_transforms(image_size=image_size),
        image_size=image_size,
        **prep_kwargs
    )

    if config['training'].get('use_weighted_sampler', False):
        print("Using weighted random sampler for class imbalance...")
    
        train_labels = train_dataset.labels
        class_counts = np.bincount(train_labels)
    
        class_weights = 1.0 / class_counts
    
        sample_weights = [class_weights[label] for label in train_labels]
    
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            num_workers=config['training']['num_workers'],
            worker_init_fn=seed_worker,
            generator=generator
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            worker_init_fn=seed_worker,
            generator=generator
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    return train_loader, val_loader, train_dataset

def setup_model_and_optimizer(
    config: dict,
    device: torch.device,
    train_dataset: DRDataset
) -> tuple:
    """Set up model, optimizer, and loss function."""
    model = create_model(
        backbone=config['model']['backbone'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    
    if config['model'].get('freeze_backbone', False):
        model.freeze_backbone()
        print('Backbone frozen - training only classifier head')

    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 1e-4)

    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=config['training'].get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    scheduler = None
    if config['training'].get('scheduler'):
        scheduler_name = config['training']['scheduler'].lower()
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs']
            )
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=3,
                factor=0.5
            )

    class_weights = None
    if config['training'].get('use_class_weights', False):
        weights = train_dataset.get_class_weights()
        num_classes = config['model']['num_classes']
        if len(weights) != num_classes:
            raise ValueError(
                f'Training labels cover {len(weights)} classes but num_classes '
                f'is {num_classes}; cannot build a class weight vector. Check '
                f'that every class is present in the training split.'
            )
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f'Using class weights: {[round(float(w), 4) for w in weights]}')

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    return model, optimizer, criterion, scheduler

def main():
    parser = argparse.ArgumentParser(description='Train DR detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint')

    args = parser.parse_args()

    config = load_config(args.config)
    print(f'Loaded configuration from {args.config}')

    if (config['training'].get('use_class_weights', False)
            and config['training'].get('use_weighted_sampler', False)):
        raise ValueError(
            'use_class_weights and use_weighted_sampler both correct for class '
            'imbalance; enabling both double-corrects it. Choose one.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    seed = config.get('seed', 42)
    set_seed(seed)
    print(f'Seeded run with seed={seed} (cuDNN set to deterministic)')

    print('\nSetting up data loaders...')
    train_loader, val_loader, train_dataset = setup_data_loaders(config)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    print('\nSetting up model and optimizer...')
    model, optimizer, criterion, scheduler = setup_model_and_optimizer(
        config, device, train_dataset
    )
    print(f'Model: {config["model"]["backbone"]}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config['training']['checkpoint_dir']
    )

    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        epoch, metrics = trainer.load_checkpoint(args.resume)
        print(f'Resumed from epoch {epoch}')

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 5)
    )

    print('\nSaving training history plot...')
    fig = plot_training_history(trainer.history)
    fig.savefig(
        Path(config['training']['checkpoint_dir']) / 'training_history.png',
        dpi=300,
        bbox_inches='tight'
    )
    # Release the figure: an abandoned figure keeps a manager reference alive,
    # which under a GUI backend would keep the process from exiting.
    plt.close(fig)

    print('\nTraining complete!')
    print(f'Best validation kappa: {trainer.best_val_kappa:.4f}')
    print(f'Best model saved to: {Path(config["training"]["checkpoint_dir"])}')


if __name__ == '__main__':
    main()

