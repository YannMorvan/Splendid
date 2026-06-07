import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss — reduces well-classified examples' contribution."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # optional class-weight tensor, shape (C,)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()
