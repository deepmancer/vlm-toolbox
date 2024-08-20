import torch
from torch import nn

from util.torch_helper import group_operation


class CoarselySupervisedLoss(nn.Module):
    def __init__(self, loss_fn, fine_to_coarse_mapping, weight=0.2, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn
        self.fine_to_coarse_label_id_mapping = fine_to_coarse_mapping
        self.weight = weight

    def forward(self, normalized_m1_features, normalized_m2_features, m1_ids, m2_ids, labels, temperature, **kwargs):
        fine_loss = self.loss_fn(
            normalized_m1_features=normalized_m1_features,
            normalized_m2_features=normalized_m2_features,
            labels=labels,
            temperature=temperature,
            **kwargs,
        )

        m2_coarse_ids = self.fine_to_coarse_label_id_mapping[m2_ids]
        grouped_m2_features = group_operation(normalized_m2_features, m2_coarse_ids, dim=0)[0]
        normalized_grouped_m2_features = grouped_m2_features / grouped_m2_features.norm(p=2, dim=-1, keepdim=True)

        m2_coarse_labels = torch.arange(grouped_m2_features.shape[0]).to(device=grouped_m2_features.device).long()
        coarse_loss = self.loss_fn(
            normalized_m1_features=normalized_grouped_m2_features,
            normalized_m2_features=normalized_grouped_m2_features,
            labels=m2_coarse_labels,
            temperature=temperature,
            **kwargs,
        )
        return fine_loss * self.weight + coarse_loss * (1 - self.weight)
