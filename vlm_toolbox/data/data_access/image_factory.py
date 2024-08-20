import numpy as np

from config.enums import DataStatus, ModalityType
from config.image_datasets import ImageDatasetConfig
from config.memory import NUM_PROCS
from config.modality import ModalityManager
from data.data_access.dataset_handler import SingleModalDatasetHandler
from data.data_access.dataset_modules import DatasetIOHandler, DatasetTransformer


class ImageHandlerFactory:
    @staticmethod
    def create_from_config(key, stage, dataset_config, to_keep_ids=None, n_shots=None):
        dataset = DatasetIOHandler.load_dataset(
            dataset_config['path'],
            dataset_config['storage_type'],
            dataset_config['split'],
        )
        image_modality = ModalityManager.get_singleton_modality(
            key=key,
            stage=stage,
            modality_type=ModalityType.IMAGE,
            source=dataset_config['source'],
        )
        image_modality.update_status(dataset_config['data_type'])
        if image_modality.identifier not in dataset.column_names:
            dataset = dataset.rename_column(dataset_config['id_column_name'], image_modality.identifier)
        
        value_col_names = [col for col in dataset.column_names if col != image_modality.identifier]
        value_col_name = value_col_names[0]
        expected_value_col_name = image_modality.get_current_values()
        expected_value_col_name = expected_value_col_name[0] if isinstance(expected_value_col_name, list) else expected_value_col_name

        if value_col_name != expected_value_col_name:
            dataset = dataset.rename_column(value_col_name, expected_value_col_name)
            value_col_names = [expected_value_col_name]


        if n_shots is not None:
            dataset = DatasetTransformer.downsample(
                dataset,
                downsample_on_col_name=image_modality.identifier,
                n_shots=n_shots,
            )

        if to_keep_ids is not None:
            unique_ids = np.unique(dataset[image_modality.identifier])
            if len(unique_ids) > len(to_keep_ids):
                to_keep_ids_set = set(to_keep_ids)
                current_ids = np.array(dataset[image_modality.identifier])
                
                mask = np.isin(current_ids, list(to_keep_ids_set))
                indices = np.where(mask)[0]
                
                mapping_array = np.full(np.max(current_ids) + 1, -1)
                mapping_array[to_keep_ids] = np.arange(len(to_keep_ids))
                mapped_ids = mapping_array[current_ids[indices]]

                dataset = dataset.select(indices)
                dataset = dataset.remove_columns(image_modality.identifier).add_column(image_modality.identifier, mapped_ids)

        return SingleModalDatasetHandler(
            dataset=dataset,
            modality=image_modality,
            config=dataset_config,
        )

    @staticmethod
    def create(
        key,
        stage,
        setup,
        data_type=DataStatus.RAW,
        to_keep_ids=None,
        n_shots=None,
    ):
        assert data_type in DataStatus.get_values()
        
        dataset_config = ImageDatasetConfig.get_config(setup, split=stage, data_type=data_type)
        return ImageHandlerFactory.create_from_config(key, stage, dataset_config, to_keep_ids=to_keep_ids, n_shots=n_shots)
