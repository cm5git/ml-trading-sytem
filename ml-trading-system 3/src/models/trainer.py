"""
Training loop for the neural network component: mixed-precision
training, LR scheduling, gradient clipping, and early stopping on
validation loss.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam


class Trainer:
    """Handles the training loop, validation, and early stopping."""

    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_labels = (all_preds > 0.5).astype(int)

        accuracy = accuracy_score(all_labels, pred_labels)
        auc = roc_auc_score(all_labels, all_preds)
        return val_loss, accuracy, auc

    def fit(self, train_loader, val_loader, checkpoint_path='/tmp/best_model.pt'):
        print("Training neural network...")
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, val_auc = self.validate(val_loader)
            self.scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_auc'].append(val_auc)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs} | "
                      f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                      f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.model.load_state_dict(torch.load(checkpoint_path))
        return history

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(X_tensor)
                probs = torch.sigmoid(outputs)
        return probs.cpu().numpy().flatten()
