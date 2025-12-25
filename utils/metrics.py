"""
Metrics and evaluation utilities for model performance
"""
import torch


def calculate_accuracy(model, loader, device, top_k=5):
    """Calculate top-k accuracy on a dataset"""
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_X)

            # Top-1 accuracy
            _, predicted = torch.max(output, 1)
            correct_top1 += (predicted == batch_y).sum().item()

            # Top-k accuracy
            _, top_k_pred = output.topk(top_k, dim=1)
            correct_topk += sum([batch_y[i] in top_k_pred[i]
                                for i in range(len(batch_y))])

            total += batch_y.size(0)

    model.train()
    return correct_top1 / total, correct_topk / total


def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    import numpy as np
    return np.exp(loss)
