"""
Training script for gunshot detection model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import json
from datetime import datetime

from .attention_model import AudioMobileNet1D
from .data_loader import GunshotDataset, get_data_loaders

class ModelTrainer:
    """Trainer for gunshot detection model."""
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        config : dict
            Training configuration
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create model
        self.model = AudioMobileNet1D(
            num_classes=config.get('num_classes', 2),
            input_channels=config.get('input_channels', 1),
            dropout_rate=config.get('dropout_rate', 0.3)
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(config.get('class_weights', [1.0, 1.0])).float().to(self.device)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Create output directory
        self.output_dir = config.get('output_dir', 'training_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}')
                
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return epoch_loss, epoch_acc, f1, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, test_loader=None):
        """Main training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1, val_pred, val_targets = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(best_model_path)
                print(f"Saved best model with Val Acc: {val_acc:.2f}%")
                
            # Early stopping check
            if epoch > 10 and self._check_early_stopping():
                print("Early stopping triggered")
                break
                
        # Load best model for final evaluation
        self.load_model(best_model_path)
        
        # Final evaluation
        print("\n" + "="*50)
        print("Final Evaluation")
        print("="*50)
        
        # Test on validation set
        val_loss, val_acc, val_f1, val_pred, val_targets = self.validate(val_loader)
        print(f"Best Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        # Generate classification report
        self._generate_classification_report(val_targets, val_pred, "Validation")
        
        # Test on test set if available
        if test_loader:
            test_loss, test_acc, test_f1, test_pred, test_targets = self.validate(test_loader)
            print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}")
            self._generate_classification_report(test_targets, test_pred, "Test")
            
        # Save training history
        self._save_training_history()
        
        # Plot training curves
        self._plot_training_curves()
        
        return self.history
    
    def evaluate(self, data_loader):
        """Evaluate model on given data loader."""
        loss, acc, f1, predictions, targets = self.validate(data_loader)
        
        print(f"Evaluation Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  F1 Score: {f1:.4f}")
        
        # Detailed classification report
        self._generate_classification_report(targets, predictions, "Evaluation")
        
        return predictions, targets
    
    def save_model(self, filepath):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load model state."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if 'config' in checkpoint:
                self.config = checkpoint['config']
                
            if 'history' in checkpoint:
                self.history = checkpoint['history']
                
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")
            
    def _check_early_stopping(self):
        """Check for early stopping conditions."""
        if len(self.history['val_loss']) < 10:
            return False
            
        # Check if validation loss hasn't improved for 10 epochs
        recent_losses = self.history['val_loss'][-10:]
        if min(recent_losses) > np.min(self.history['val_loss'][:-10]):
            return True
            
        return False
    
    def _generate_classification_report(self, targets, predictions, dataset_name):
        """Generate and print classification report."""
        from sklearn.metrics import classification_report, confusion_matrix
        
        print(f"\n{classification_report(targets, predictions, target_names=['Non-Gunshot', 'Gunshot'])}")
        
        # Plot confusion matrix
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Gunshot', 'Gunshot'],
                   yticklabels=['Non-Gunshot', 'Gunshot'])
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(self.output_dir, f'confusion_matrix_{dataset_name.lower()}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _save_training_history(self):
        """Save training history to file."""
        history_df = pd.DataFrame(self.history)
        history_path = os.path.join(self.output_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        
        # Save config
        config_path = os.path.join(self.output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def _plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        curves_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    # Configuration
    config = {
        # Model parameters
        'num_classes': 2,
        'input_channels': 1,
        'dropout_rate': 0.3,
        
        # Training parameters
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        
        # Data parameters
        'data_dir': 'data/audio_samples',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        
        # Class weights (handle imbalance)
        'class_weights': [1.0, 2.0],  # Gunshot class gets more weight
        
        # Output
        'output_dir': f'training_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )
    
    # Create and run trainer
    trainer = ModelTrainer(config)
    history = trainer.train(train_loader, val_loader, test_loader)
    
    # Save final model
    final_model_path = os.path.join(trainer.output_dir, 'final_model.pth')
    trainer.save_model(final_model_path)
    
    print(f"\nTraining completed. Results saved to {trainer.output_dir}")


if __name__ == "__main__":
    main()
