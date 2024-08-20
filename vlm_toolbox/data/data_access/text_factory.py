from datasets import Dataset

from config.enums import DataStatus, ModalityType
from config.modality import ModalityManager
from data.data_access.dataset_handler import SingleModalDatasetHandler


class TextHandlerFactory:
    @staticmethod
    def create_from_df(key, stage, labels_df, config={}):
        text_modality = ModalityManager.get_singleton_modality(
            key=key,
            stage=stage,
            modality_type=ModalityType.TEXT,
        )
        text_modality.update_status(DataStatus.RAW)
        dataset = Dataset.from_pandas(labels_df)

        return SingleModalDatasetHandler(
            dataset,
            modality=text_modality,
            config=config,
        )
