import torch
import torch.nn as nn
import torch.nn.functional as F

class EnlargedLargeMarginLoss(nn.Module):
    def __init__(self, labels_sample_count, max_m=0.5, lamda=1.0, weight=None, s=30, reduction='mean', **kwargs):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(labels_sample_count.float()))
        m_list = m_list * (max_m / torch.max(m_list))
        self.register_buffer('m_list', m_list)
        self.s = s
        self.weight = weight
        self.lamda = lamda
        self.reduction = reduction

    def forward(self, normalized_m1_features, normalized_m2_features, labels, temperature, **kwargs):
        device = normalized_m1_features.device
        logits = temperature * torch.matmul(normalized_m1_features, normalized_m2_features.t())

        batch_size, num_classes = logits.shape
        index = torch.zeros_like(logits, dtype=torch.bool, device=device)
        index.scatter_(1, labels.view(-1, 1), 1)
        index_float = index.type(torch.float32)

        logits_clone = logits.clone()
        logits_clone[index] = torch.finfo(logits_clone.dtype).min
        max_other_class = logits_clone.argmax(dim=1)
        index2 = torch.zeros_like(logits, dtype=torch.bool, device=device)
        index2.scatter_(1, max_other_class.view(-1, 1), 1)
        index_float2 = index2.type(torch.float32)

        batch_m1 = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1)).view(-1, 1)
        batch_m2 = torch.matmul(self.m_list[None, :], index_float2.transpose(0, 1)).view(-1, 1)

        logits_m = logits - batch_m1 + batch_m2 * self.lamda
        output = torch.where(index, logits_m, logits)

        loss = F.cross_entropy(self.s * output, labels, weight=self.weight, reduction=self.reduction)
        return loss
