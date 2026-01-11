import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np

from src.utils.metrics import calculate_metrics

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler]=None,
        checkpoint_dir: str='checkpoints'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(all_labels)

        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )

        return {
            'loss': epoch_loss,
            'acc': metrics['accuracy'],
            'metrics': metrics
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int=5
    ):
        """Train the model for multiple epochs."""
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            train_metrics = self.train_epoch(train_loader)
            print(
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Train Acc: {train_metrics["acc"]:.4f}'
            )

            val_metrics = self.validate(val_loader)
            print(
                f'Val Loss: {val_metrics["loss"]:.4f}, '
                f'Val Acc: {val_metrics["acc"]:.4f}'
            )

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            if self.scheduler is not None:
                self.scheduler.step()

            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)

                print(f'✓ Saved best model (acc: {val_metrics["acc"]:.4f})')

                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

        print('\nTraining complete!')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, Any]
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint.get('history', self.history)

        return checkpoint['epoch'], checkpoint['metrics']
