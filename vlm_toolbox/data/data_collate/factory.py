import torch

from config.enums import ModalityType
from data.data_collate.multimodal_collator import MultiModalDataCollator
from data.data_collate.singlemodal_collators import (
    SingleModalDataCollator,
    TextDataCollator,
)


class DataCollatorFactory:
    @staticmethod
    def create_singlemodal_collator(dataset_handler, aggregate_samples=False, processor=None):
        processor = torch.stack if processor is None else processor
        if dataset_handler.get_type() == ModalityType.TEXT:
            dataset_handler.build_index()

        identifier = dataset_handler.get_identifier()
        if dataset_handler.get_type() == ModalityType.IMAGE:
            return SingleModalDataCollator(identifier, processor=processor, aggregate_samples=aggregate_samples)
        elif dataset_handler.get_type() == ModalityType.TEXT:
            return TextDataCollator(identifier, processor=processor, aggregate_samples=aggregate_samples)
        else:
            return ValueError
    
    @staticmethod
    def create_multimodal_collator(
        m1_id_m2_id_adj_matrix,
        m1_dataset_handler,
        m2_dataset_handler,
        m1_processor=None,
        m2_processor=None,
        collate_all_m2_samples=False,
        aggregate_m1_samples=False,
        aggregate_m2_samples=True,
    ):
        m1_collator = DataCollatorFactory.create_singlemodal_collator(
            m1_dataset_handler,
            processor=m1_processor,
            aggregate_samples=aggregate_m1_samples,
        )
        m2_collator = DataCollatorFactory.create_singlemodal_collator(
            m2_dataset_handler,
            processor=m2_processor,
            aggregate_samples=aggregate_m2_samples,
        )
        if collate_all_m2_samples:
            m2_examples = m2_dataset_handler.get_dataset(return_pt=True, keep_necessary_cols_only=True)
            m2_retriever_fn = lambda m2_ids: m2_examples
        else:
            m2_retriever_fn = lambda m2_ids: m2_dataset_handler.filter(
                m2_ids,
                return_pt=True,
                keep_necessary_cols_only=True,
            )

        return MultiModalDataCollator(m1_id_m2_id_adj_matrix, m1_collator, m2_collator, m2_retriever_fn)
