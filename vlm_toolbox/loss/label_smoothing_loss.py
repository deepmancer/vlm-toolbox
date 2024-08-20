import torch
from timm.loss import LabelSmoothingCrossEntropy
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, is_symmetric=False, **kwargs):
        super().__init__()
        self.loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.is_symmetric = is_symmetric

    def forward(self, normalized_m1_features, normalized_m2_features, labels, temperature, **kwargs):
        logits = temperature * torch.matmul(normalized_m1_features, normalized_m2_features.t())
        if self.is_symmetric:
            loss_m1 = self.loss_fn(logits, labels)
            loss_m2 = self.loss_fn(logits.t(), labels.t())
            return (loss_m1 + loss_m2) / 2.0

        return self.loss_fn(logits, labels)
