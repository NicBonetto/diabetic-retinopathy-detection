import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import sys
sys.path.append(str(Path(__file__).parent))

from src.core.data import DRDataset
from src.core.classifiers import create_model
from src.core.trainer import Trainer
from src.utils.visualizations import plot_training_history

def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def setup_data_loaders(config: dict) -> tuple:
    """Set up training and validation data loaders."""
    train_dataset = DRDataset(
        data_dir=config['data']['train_dir'],
        labels_file=config['data']['train_labels'],
        transform=DRDataset.get_train_transforms(image_size=tuple(config['data']['image_size'])),
        image_size=tuple(config['data']['image_size'])
    )

    val_dataset = DRDataset(
        data_dir=config['data']['val_dir'],
        labels_file=config['data']['val_labels'],
        transform=DRDataset.get_val_transforms(image_size=tuple(config['data']['image_size'])),
        image_size=tuple(config['data']['image_size'])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset

def setup_model_and_optimizer(config: dict, device: torch.device) -> tuple:
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

    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion, scheduler

def main():
    parser = argparse.ArgumentParser(description='Train DR detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint')

    args = parser.parse_args()

    config = load_config(args.config)
    print(f'Loaded configuration from {args.config}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get('seed', 42))

    print('\nSetting up data loaders...')
    train_loader, val_loader, train_dataset = setup_data_loaders(config)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    print('\nSetting up model and optimizer...')
    model, optimizer, criterion, scheduler = setup_model_and_optimizer(config, device)
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
        Path(config['training']['checkpoint_dur']) / 'training_history.png',
        dpi=300,
        bbox_inches='tight'
    )

    print('\nTraining complete!')
    print(f'Best validation accuracy: {trainer.best_val_acc:.4f}')
    print(f'Best model saved to: {Path(config["training"]["checkpoint_dir"])}')


if __name__ == '__main__':
    main()

