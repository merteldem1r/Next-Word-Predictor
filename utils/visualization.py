"""
Visualization utilities for training metrics
"""
import os
import matplotlib.pyplot as plt


def plot_metrics(history, results_dir, figsize=(10, 6), dpi=150):
    """Generate and save training visualization plots"""
    os.makedirs(results_dir, exist_ok=True)

    epochs_range = range(1, len(history['train_loss']) + 1)

    # Loss plot
    plt.figure(figsize=figsize)
    plt.plot(epochs_range, history['train_loss'],
             'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'],
             'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/loss_curve.png", dpi=dpi)
    plt.close()
    print(f"Saved: {results_dir}/loss_curve.png")

    # Top-1 Accuracy plot
    plt.figure(figsize=figsize)
    plt.plot(epochs_range, history['train_acc_top1'],
             'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs_range, history['val_acc_top1'],
             'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy (Top-1)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/accuracy_top1_curve.png", dpi=dpi)
    plt.close()
    print(f"Saved: {results_dir}/accuracy_top1_curve.png")

    # Top-5 Accuracy plot
    plt.figure(figsize=figsize)
    plt.plot(epochs_range, history['train_acc_top5'],
             'b-', label='Train Top-5 Accuracy', linewidth=2)
    plt.plot(epochs_range, history['val_acc_top5'], 'r-',
             label='Validation Top-5 Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy (Top-5)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/accuracy_top5_curve.png", dpi=dpi)
    plt.close()
    print(f"Saved: {results_dir}/accuracy_top5_curve.png")

    # Perplexity plot
    plt.figure(figsize=figsize)
    plt.plot(epochs_range, history['train_perplexity'],
             'b-', label='Train Perplexity', linewidth=2)
    plt.plot(epochs_range, history['val_perplexity'],
             'r-', label='Validation Perplexity', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Training and Validation Perplexity',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/perplexity_curve.png", dpi=dpi)
    plt.close()
    print(f"Saved: {results_dir}/perplexity_curve.png")


def plot_loss_curve(history, results_dir):
    """Plot only loss curve"""
    os.makedirs(results_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'],
             'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'],
             'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/loss_curve.png", dpi=150)
    plt.close()
