from torch import nn


class WeightedL1Loss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean', **kwargs):
        super().__init__()
        self.weight = weight
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, normalized_m1_features, normalized_m2_features, **kwargs):
        loss = self.loss_fn(normalized_m1_features, normalized_m2_features) * self.weight
        return loss
