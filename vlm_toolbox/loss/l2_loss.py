import torch
from torch import nn


class WeightedL2Loss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, normalized_m1_features, normalized_m2_features, **kwargs):
        m1_expanded = normalized_m1_features.unsqueeze(1)
        
        squared_diff = (m1_expanded - normalized_m2_features) ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=2)
        if self.reduction == 'mean':
            mean_squared_diff = torch.mean(sum_squared_diff, dim=1)
        elif self.reduction == 'sum':
            mean_squared_diff = torch.sum(sum_squared_diff, dim=1)
        else:
            raise ValueError("Unsupported reduction type. Use 'mean' or 'sum'.")
        
        loss = mean_squared_diff * self.weight
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss
