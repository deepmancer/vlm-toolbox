import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction='mean', is_symmetric=True, is_multi_class=False, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.is_multi_class = is_multi_class
        self.is_symmetric = is_symmetric
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction) if is_multi_class else nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, normalized_m1_features, normalized_m2_features, labels, temperature, **kwargs):
        logits = temperature * torch.matmul(normalized_m1_features, normalized_m2_features.t())
        if self.is_symmetric:
            loss_m1 = self.loss_fn(logits, labels)
            loss_m2 = self.loss_fn(logits.t(), labels.t())
            return (loss_m1 + loss_m2) / 2.0

        return self.loss_fn(logits, labels)
