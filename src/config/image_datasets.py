import os

from config.base import BaseConfig
from config.enums import DataStatus, ImageDatasets, StorageType
from config.path import IMAGES_TEMPLATE_PATH, IMAGE_EMBEDS_TEMPLATE_PATH


class ImageDatasetConfig(BaseConfig):
    config = {
        ImageDatasets.IMAGENET_1K: {
            'splits': ['train', 'validation'],
            DataStatus.RAW: {
                'path': IMAGES_TEMPLATE_PATH.format(dataset_name=ImageDatasets.IMAGENET_1K),
                'type': StorageType.DISK,
            },
            DataStatus.EMBEDDING: {
                'path': IMAGE_EMBEDS_TEMPLATE_PATH.format(dataset_name=ImageDatasets.IMAGENET_1K),
                'type': StorageType.DISK,
            },
            'id_col': 'label',
        },
        ImageDatasets.MSCOCO_CAPTIONS: {
            'splits': ['train', 'validation'],
            DataStatus.RAW: {
                'path': 'HuggingFaceM4/COCO',
                'type': StorageType.HUGGING_FACE,
            },
            'id_col': 'imgid',
        },
        ImageDatasets.STANFORD_CARS: {
            'splits': ['train', 'test'],
            DataStatus.RAW: {
                'path': IMAGES_TEMPLATE_PATH.format(dataset_name=ImageDatasets.STANFORD_CARS),
                'type': StorageType.IMAGE_FOLDER,
            },
            DataStatus.EMBEDDING: {
                'path': IMAGE_EMBEDS_TEMPLATE_PATH.format(dataset_name=ImageDatasets.STANFORD_CARS),
                'type': StorageType.DISK,
            },
            'id_col': 'label',
        },
        ImageDatasets.FOOD101: {
            'splits': ['train', 'validation'],        
            DataStatus.RAW: {
                'path': 'food101',
                'type': StorageType.HUGGING_FACE,
            },
            DataStatus.EMBEDDING: {
                'path': IMAGE_EMBEDS_TEMPLATE_PATH.format(dataset_name=ImageDatasets.FOOD101),
                'type': StorageType.DISK,
            },
            'id_col': 'label',
        },
        ImageDatasets.CIFAR100: {
            'splits': ['train', 'test'],
            DataStatus.RAW: {
                'path': 'cifar100',
                'type': StorageType.HUGGING_FACE,
            },
            DataStatus.EMBEDDING: {
                'path': IMAGE_EMBEDS_TEMPLATE_PATH.format(dataset_name=ImageDatasets.CIFAR100),
                'type': StorageType.DISK,
                
            },
            'id_col': 'fine_label',
        },
        ImageDatasets.INATURALIST: {
            'splits': ['train', 'validation'],
            DataStatus.RAW: {
                'path': IMAGES_TEMPLATE_PATH.format(dataset_name=ImageDatasets.INATURALIST),
                'type': StorageType.DISK,
            },
            DataStatus.EMBEDDING: {
                'path': IMAGE_EMBEDS_TEMPLATE_PATH.format(dataset_name=ImageDatasets.INATURALIST),
                'type': StorageType.DISK,
            },
            'id_col': 'name',
        },
    }
    @staticmethod
    def validate_dataset_name(dataset_name):
        if dataset_name not in ImageDatasetConfig.config:
            raise ValueError(f"Dataset '{dataset_name}' configuration not found.")
    
    @staticmethod
    def validate_split(dataset_name, split):
        if split not in ImageDatasetConfig.config[dataset_name]['splits']:
            raise ValueError(f"Split '{split}' not available for dataset '{dataset_name}'.")
    
    @staticmethod
    def validate_data_type(dataset_name, data_type):
        dataset_config = ImageDatasetConfig.config[dataset_name]
        if data_type not in dataset_config:
            raise ValueError(f"Data type '{data_type}' not available for dataset '{dataset_name}'.")
    
    @staticmethod
    def format_path(data_config, split, data_type, backbone_name=None, source=None):
        path_template = data_config['path']
        if data_type == DataStatus.EMBEDDING:
            return path_template.format(backbone_name=backbone_name, source=source, split=split)
        if data_config['type'] == StorageType.HUGGING_FACE:
            return path_template
        return path_template.format(split=split)
    
    @staticmethod
    def ensure_path_exists(path, storage_type):
        if storage_type in [StorageType.DISK, StorageType.IMAGE_FOLDER] and not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist.")

    @staticmethod
    def get_config(setup, split, data_type):
        dataset_name = setup.dataset_name.lower()
        backbone_name, source = setup.get_backbone_name() , setup.get_source()

        ImageDatasetConfig.validate_dataset_name(dataset_name)
        ImageDatasetConfig.validate_split(dataset_name, split)
        ImageDatasetConfig.validate_data_type(dataset_name, data_type)
        
        dataset_config = ImageDatasetConfig.config[dataset_name]
        data_config = dataset_config[data_type]
        path = ImageDatasetConfig.format_path(data_config, split, data_type, backbone_name, source)

        ImageDatasetConfig.ensure_path_exists(path, data_config['type'])
        
        return {
            'dataset_name': dataset_name,
            'split': split,
            'source': source,
            'data_type': data_type,
            'storage_type': data_config['type'],
            'path': path,
            'id_column_name': dataset_config['id_col'],
            'template_path': {
                DataStatus.RAW: dataset_config[DataStatus.RAW]['path'],
                DataStatus.EMBEDDING: dataset_config[DataStatus.EMBEDDING]['path'],
            }
        }
