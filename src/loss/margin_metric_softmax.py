import torch
from torch import nn


class MarginMetricLoss(nn.Module):
    def __init__(self, adaptive_weight=0.3, reduction='mean', **kwargs):
        super().__init__()
        self.adaptive_weight = adaptive_weight
        self.reduction = reduction

    def forward(self, normalized_m1_features, normalized_m2_features, labels, temperature, **kwargs):
        assert len(labels.shape) == 1, 'Multi-class is not supported!'

        m1_m2_similarity = torch.matmul(normalized_m1_features, normalized_m2_features.t())
        true_similarities = m1_m2_similarity[torch.arange(normalized_m1_features.size(0)), labels]

        numerator = torch.exp(true_similarities * temperature)
        m2_m2_similarity = torch.matmul(normalized_m2_features, normalized_m2_features.t())
        adjustment = self.adaptive_weight * (1 - m2_m2_similarity[labels, :])
        adjusted_similarity = m1_m2_similarity + adjustment
        scaled_adjusted_similarity = adjusted_similarity * temperature
        
        denominator = torch.exp(scaled_adjusted_similarity).sum(dim=1)
        loss_per_example = -torch.log(numerator / denominator)

        if self.reduction == 'mean':
            return loss_per_example.mean()
        elif self.reduction == 'sum':
            return loss_per_example.sum()
        else:
            return loss_per_example
