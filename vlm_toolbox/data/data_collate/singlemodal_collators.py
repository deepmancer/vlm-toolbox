import torch

from data.data_collate.base import BaseDataCollator


class SingleModalDataCollator(BaseDataCollator):
    def __init__(self, identifier, processor=torch.stack, aggregate_samples=False, id_only=False):
        self.identifier = identifier
        self.processor = processor
        self.aggregate_samples = aggregate_samples
        self.id_only = id_only
        self.label_names = [self.identifier]
        self.return_tensors = 'pt'

    def get_aggregate_samples(self):
        return self.aggregate_samples
        
    def set_label_names(self, label_names):
        self.label_names = label_names

    def get_label_names(self):
        return self.label_names

    def get_ids(self, batches):
        return torch.tensor([batch[self.identifier] for batch in batches], dtype=torch.long)

    def get_batch_values(self, batches):
        if self.id_only:
            return {}

        values = {}
        for key in batches[0].keys():
            if key == self.identifier and len(batches[0].keys()) > 1:
                continue
            result = self.processor([batch[key] for batch in batches])
            if not isinstance(result, dict):
                values[key] = result
            else:
                for nested_key in result.keys():
                    values[nested_key] = result[nested_key]
        return values

    def compute_uniqueness(self, ids):
        unique_ids, inverse_indices = None, None
        if self.aggregate_samples:
            unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
            if unique_ids.shape[0] == ids.shape[0]:
                unique_ids, inverse_indices = None, None
        return unique_ids, inverse_indices

    def __call__(self, examples):
        ids = self.get_ids(examples)
        batch_values = self.get_batch_values(examples)
        unique_ids, inverse_indices = self.compute_uniqueness(ids)

        return {
            'ids': ids,
            'unique_ids': unique_ids,
            'inverse_index': inverse_indices,
            'features': batch_values
        }

    def __str__(self):
        return f"{self.__class__.__name__}({self.identifier}, {self.label_names})"

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items() if key in self.__dict__)
        return f"{class_name}({attributes})"

class TextDataCollator(SingleModalDataCollator):
    def __init__(self, identifier, processor=torch.stack, aggregate_samples=False, id_only=False):
        super().__init__(identifier, processor, aggregate_samples, id_only)

    def sort_text_data(self, label_ids, text_batch_values):
        sorted_indices = label_ids.argsort()
        sorted_label_ids = label_ids[sorted_indices]
        text_batch_values = {key: value[sorted_indices] for key, value in text_batch_values.items()}
        return sorted_label_ids, text_batch_values

    def __call__(self, examples):
        if self.id_only:
            return super().__call__(examples)

        ids = self.get_ids(examples)
        batch_values = self.get_batch_values(examples)
        ids, batch_values = self.sort_text_data(ids, batch_values)

        unique_ids, inverse_indices = self.compute_uniqueness(ids)

        return {
            'ids': ids,
            'unique_ids': unique_ids,
            'inverse_index': inverse_indices,
            'features': batch_values
        }
