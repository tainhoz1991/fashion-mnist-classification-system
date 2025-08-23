import mlflow
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MLflowTracker:
    def __init__(self, model, classes):
        self.model = model
        self.classes = classes
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log metrics for an epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            step=epoch,
        )

    def log_confusion_matrix(self, val_loader, device):
        """Generate and log confusion matrix."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # Save and log
        plt.savefig("confusion_matrix.png")
        # mlflow only log image
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

    def log_training_curves(self):
        """Generate and log training curves."""
        plt.figure(figsize=(12, 5))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label="Train Accuracy")
        plt.plot(self.val_accs, label="Validation Accuracy")
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_curves.png")
        # mlflow only log image
        mlflow.log_artifact("training_curves.png")
        plt.close()