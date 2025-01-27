
from config.base import BaseConfig
from config.enums import (
    DataStatus,
    Modalities,
    ModalityType,
    Sources,
    Stages,
    Trainers,
)


class Modality:
    def __init__(
        self,
        modality_type,
        source,
        identifier,
        raw_keys=None,
        preprocessed_keys=None,
        embedding_key=None,
        status=DataStatus.RAW,
        requires_grad=True,
        perform_augmentation=False,
        requires_preprocess=True,
        key=None,
        stage=None,
    ):
        default_source_config = SourcesModalityConfig.get_config(source)[modality_type]
        
        self.modality_type = modality_type
        self.source = source
        self.identifier = identifier
        self.raw_keys = raw_keys or default_source_config[DataStatus.RAW]
        self.preprocessed_keys = preprocessed_keys or default_source_config[DataStatus.PREPROCESSED]
        self.embedding_key = embedding_key or default_source_config[DataStatus.EMBEDDING]
        self.status = DataStatus.get(status)
        self.initial_status = self.status
        self.requires_preprocess = requires_preprocess
        self.perform_augmentation = perform_augmentation
        self.requires_grad = requires_grad
        self.key = key
        self.stage = stage
        self.initialized = True

    def reset_status(self):
        self.status = self.initial_status

    def set_source(self, source):
        self.source = source

    def get_type(self):
        return self.modality_type

    def is_raw(self):
        return self.status == DataStatus.RAW

    def is_preprocessed(self):
        return self.status == DataStatus.PREPROCESSED or self.status == DataStatus.EMBEDDING

    def is_embedded(self):
        return self.status == DataStatus.EMBEDDING

    def get_identifier(self):
        return self.identifier

    def set_identifier(self, identifier):
        self.identifier = identifier

    def get_raw_keys(self):
        return self.raw_keys

    def set_raw_keys(self, raw_keys):
        self.raw_keys = raw_keys

    def get_preprocessed_keys(self, output=True):
        return self.preprocessed_keys['output' if output else 'input']

    def set_preprocessed_keys(self, preprocessed_keys):
        self.preprocessed_keys = preprocessed_keys

    def get_embedding_keys(self, output=True):
        return self.embedding_key['output' if output else 'input']

    def set_embedding_keys(self, embedding_key):
        self.embedding_key = embedding_key

    def get_requires_grad(self):
        return self.requires_grad

    def set_requires_grad(self, requires_grad):
        self.requires_grad = requires_grad

    def get_requires_preprocess(self):
        return self.requires_preprocess

    def set_requires_preprocess(self, requires_preprocess):
        self.requires_preprocess = requires_preprocess

    def get_perform_augmentation(self):
        return self.perform_augmentation

    def set_perform_augmentation(self, perform_augmentation):
        self.perform_augmentation = perform_augmentation

    def get_key(self):
        return self.key

    def get_stage(self):
        return self.stage

    def set_key(self, key):
        self.key = key

    def set_stage(self, stage):
        self.stage = stage

    def update_status(self, status_str):
        if status_str is not None:
            self.status = DataStatus.get(status_str)
            ModalityManager._instances[self.key][self.stage] = self

    def get_status(self):
        return self.status

    def get_next_status(self):
        if self.status == DataStatus.RAW and self.requires_preprocess:
            return DataStatus.PREPROCESSED
        if self.status == DataStatus.RAW and not self.requires_preprocess:
            return DataStatus.EMBEDDING
        elif self.status in [DataStatus.PREPROCESSED, DataStatus.EMBEDDING]:
            return DataStatus.EMBEDDING
        else:
            raise ValueError('status not supported')

    def get_current_values(self):
        if self.status == DataStatus.RAW:
            return self.get_raw_keys()
        elif self.status == DataStatus.PREPROCESSED:
            return self.get_preprocessed_keys(output=True)
        elif self.status == DataStatus.EMBEDDING:
           return self.get_embedding_keys(output=True)
        else:
            raise ValueError('status not supported')

    def get_next_values(self):
        if self.status == DataStatus.RAW and self.requires_preprocess:
            return self.get_preprocessed_keys(output=False)
        if self.status == DataStatus.RAW and not self.requires_preprocess:
            return self.get_embedding_keys(output=False)
        elif self.status == DataStatus.PREPROCESSED:
            return self.get_embedding_keys(output=False)
        elif self.status == DataStatus.EMBEDDING:
           return self.get_embedding_keys(output=True)
        else:
            raise ValueError('status not supported')

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __repr__(self):
        attributes = []
        attribute_list = [
            'modality_type', 'identifier', 'raw_keys', 'preprocessed_keys', 'key', 'stage',
            'embedding_key', 'status', 'requires_grad', 'requires_preprocess', 'perform_augmentation'
        ]
        
        for attr in attribute_list:
            value = getattr(self, attr)
            if value is not None:
                if isinstance(value, str):
                    value = f"'{value}'"
                elif isinstance(value, bool):
                    value = str(value).lower()
                attributes.append(f"{attr}={value}")
    
        return f"{self.__class__.__name__}({', '.join(attributes)})"
    
    
class ModalityManager:
    config = {
        ModalityType.TEXT: {
            'identifier': 'label_id',
        },
        ModalityType.IMAGE: {
            'identifier': 'class_id',
        },
    }
    _instances = {Modalities.M1: {}, Modalities.M2: {}}

    @staticmethod
    def flush():
        ModalityManager._instances = {Modalities.M1: {}, Modalities.M2: {}}

    @staticmethod
    def reset():
        for modality_key, stages in ModalityManager._instances.items():
            for stage, modality_instance in stages.items():
                modality_instance.status = modality_instance.initial_status
    
    @staticmethod
    def get_singleton_modality(key, stage, modality_type=ModalityType.IMAGE, source=Sources.OPEN_AI, **kwargs):
        if key not in [Modalities.M1, Modalities.M2]:
            raise ValueError("Key must be Modalities.M1 or Modalities.M2")

        if stage not in Stages.get_values():
            raise ValueError(f"stage must be in {Stages.get_values()}")

        if stage not in ModalityManager._instances[key]:
            modality_instance = ModalityManager.get_modality(
                modality_type,
                source=source,
                key=key,
                stage=stage,
                **kwargs,
            )
            ModalityManager._instances[key][stage] = modality_instance
    
        return ModalityManager._instances[key][stage]

    @staticmethod
    def get_modality(modality_type, source=Sources.HUGGINGFACE, **kwargs):
        modality_identifier = ModalityManager.config[modality_type]['identifier']
        modality_instance = Modality(
            modality_type=modality_type,
            source=source,
            identifier=modality_identifier,
            **kwargs,
        )
        return modality_instance

    @staticmethod
    def initialize_modalities(trainer_name, source):
        ModalityManager._instances = {Modalities.M1: {}, Modalities.M2: {}}
        ModalityManager._instances = TrainersModalityConfig.get_config(trainer_name, source)

class TrainersModalityConfig(BaseConfig):
    config = {
        Trainers.CLIP: {
            Stages.TRAIN: {
                Modalities.M1: {'modality_type': ModalityType.IMAGE, 'perform_augmentation': False},
                Modalities.M2: {'modality_type': ModalityType.TEXT},
            },
            Stages.EVAL: {
                Modalities.M1: {'modality_type': ModalityType.IMAGE},
                Modalities.M2: {'modality_type': ModalityType.TEXT},
            },
        },
        Trainers.COOP: {
            Stages.TRAIN: {
                Modalities.M1: {
                    'modality_type': ModalityType.IMAGE,
                    'requires_grad': False,
                    'perform_augmentation': False,
                },
                Modalities.M2: {
                    DataStatus.RAW: ['label_id'],
                    DataStatus.EMBEDDING: {'input': ['label_id']},
                    'requires_preprocess': False,
                    'modality_type': ModalityType.TEXT,
                },
            },
            Stages.EVAL: {
                Modalities.M1: {'modality_type': ModalityType.IMAGE},
                Modalities.M2: {
                    DataStatus.RAW: ['label_id'],
                    DataStatus.EMBEDDING: {'input': ['label_id']},
                    'requires_preprocess': False,
                    'modality_type': ModalityType.TEXT,
                },
            },
        },
    }

    @staticmethod
    def get_config(trainer_name, source):
        def update_with_defaults(custom, default):
            if isinstance(default, dict):
                for key, value in default.items():
                    if key in custom:
                        if isinstance(custom[key], dict) and isinstance(value, dict):
                            update_with_defaults(custom[key], value)
                    else:
                        custom[key] = value

        trainer_config = TrainersModalityConfig.config.get(trainer_name)
        SourcesModalityConfig.get_config(source)

        imputed_config = {Modalities.M1: {}, Modalities.M2: {}}
        
        for stage_key, stage_value in trainer_config.items():
            for modality_key, modality_value in stage_value.items():
                modality_type = modality_value['modality_type']
                default_modality_config = SourcesModalityConfig.get_config(source)[modality_type]
                
                update_with_defaults(modality_value, default_modality_config)
                

                raw_keys = modality_value[DataStatus.RAW]
                preprocessed_keys = modality_value[DataStatus.PREPROCESSED]
                embedding_key = modality_value[DataStatus.EMBEDDING]
                
                imputed_config[modality_key][stage_key] = ModalityManager.get_singleton_modality(
                    key=modality_key,
                    stage=stage_key,
                    modality_type=modality_type,
                    source=source,
                    raw_keys=raw_keys,
                    preprocessed_keys=preprocessed_keys,
                    embedding_key=embedding_key,
                    requires_preprocess=modality_value.get('requires_preprocess', True),
                    perform_augmentation=modality_value.get('perform_augmentation', False),
                    requires_grad=modality_value.get('requires_grad', stage_key == Stages.TRAIN)
                )
            
        return imputed_config


class SourcesModalityConfig(BaseConfig):
    config = {
        Sources.HUGGINGFACE: {
            ModalityType.IMAGE: {
                DataStatus.RAW: ['image'],
                DataStatus.PREPROCESSED: {
                    'input': ['image'],
                    'output': ['pixel_values'],
                },
                DataStatus.EMBEDDING: {
                    'input': ['pixel_values'],
                    'output': ['image_embeds'],
                },
                'requires_preprocess': True,
            },
            ModalityType.TEXT: {
                DataStatus.RAW: ['label'],
                DataStatus.PREPROCESSED: {
                    'input': ['label'],
                    'output': ['input_ids', 'attention_mask'],
                },
                DataStatus.EMBEDDING: {
                    'input': ['input_ids', 'attention_mask'],
                    'output': ['text_embeds'],
                },
                'requires_preprocess': True,
            },
        },
        Sources.OPEN_AI: {
            ModalityType.IMAGE: {
                DataStatus.RAW: ['image'],
                DataStatus.PREPROCESSED: {
                    'input': ['image'],
                    'output': ['pixel_values'],
                },
                DataStatus.EMBEDDING: {
                    'input': ['pixel_values'],
                    'output': ['image_embeds'],
                },
                'requires_preprocess': True,
            },
            ModalityType.TEXT: {
                DataStatus.RAW: ['label'],
                DataStatus.PREPROCESSED: {
                    'input': ['label'],
                    'output': ['input_ids'],
                },
                DataStatus.EMBEDDING: {
                    'input': ['input_ids'],
                    'output': ['text_embeds'],
                },
                'requires_preprocess': True,
            },
        },
    }

    @staticmethod
    def get_config(source=Sources.HUGGINGFACE):
        assert source in SourcesModalityConfig.config.keys(), f'Source not available! Choose between: {list(SourcesModalityConfig.config.keys())}'
        return SourcesModalityConfig.config[source]
