import torch

from config.enums import Modalities
from data.data_collate.base import BaseDataCollator


class MultiModalDataCollator(BaseDataCollator):
    def __init__(self, m1_id_m2_id_adj_matrix, m1_collator, m2_collator, m2_retriever_fn):
        super().__init__()
        self.m1_id_m2_id_adj_matrix = m1_id_m2_id_adj_matrix
        self.m1_collator = m1_collator
        self.m2_collator = m2_collator
        self.m2_retriever_fn = m2_retriever_fn
        self.is_multi_class = torch.any(torch.sum(m1_id_m2_id_adj_matrix == 1, dim=1) > 1).item()

    @classmethod
    def get_label_names(cls):
        return ['labels']
    
    def get_labels(self, m1_ids, m2_ids): 
        labels_matrix = self.m1_id_m2_id_adj_matrix[m1_ids][:, m2_ids].squeeze()        
        if self.is_multi_class:
            return labels_matrix.float()     
        else:
            flat_scaled_indices = torch.argmax(labels_matrix, dim=1).long()
            return flat_scaled_indices

    def get_m2_ids_to_retrieve(self, m1_ids):
        m2_mask = self.m1_id_m2_id_adj_matrix[m1_ids].any(dim=0)
        m2_ids_to_retrieve = torch.unique(torch.where(m2_mask)[0])
        return m2_ids_to_retrieve            

    def __call__(self, m1_examples, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        m1_batch = self.m1_collator(m1_examples)
        m2_ids_to_retrieve = self.get_m2_ids_to_retrieve(m1_batch['ids'])
        m2_examples = self.m2_retriever_fn(m2_ids_to_retrieve.tolist())
        m2_batch = self.m2_collator(m2_examples)

        ground_truth_labels = self.get_labels(
            m1_batch['unique_ids'] if m1_batch['unique_ids'] is not None and self.m1_collator.get_aggregate_samples() else m1_batch['ids'],
            m2_batch['unique_ids'] if self.is_multi_class and self.m2_collator.get_aggregate_samples() else m2_batch['ids'],
        )
        return {
            'labels': ground_truth_labels,
            'is_multi_class': self.is_multi_class,
            Modalities.M1: m1_batch,
            Modalities.M2: m2_batch,
        }    
    
    def __str__(self):
        return f"{self.__class__.__name__}(m1_collator={str(self.m1_collator)}, m2_collator={str(self.m2_collator)}, is_multi_class={self.is_multi_class})"

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items() if key in self.__dict__)
        return f"{class_name}({attributes})"
