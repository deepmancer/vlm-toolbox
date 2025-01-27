from config.backbones import BackboneURLConfig
from config.base import BaseConfig
from config.enums import (
    CLIPBackbones,
    LossType,
    ModelType,
    Sources,
    Stages,
    Trainers,
)
from config.loss import LossConfig, TrainerLoss, TrainersLossConfig
from config.modality import TrainersModalityConfig
from config.path import EXPERIMENTS_MODEL_DIR
from config.soft_prompting import SoftPromptingConfig
from config.transforms import TransformationConfigManager


class ModelConfig:
    path_config = {
        ModelType.ZERO_SHOT: '{model_type}/{dataset_name}/{backbone_name}/{source}/{trainer_name}/',
        ModelType.FEW_SHOT: '{model_type}/{dataset_name}/{backbone_name}/{source}/{trainer_name}/{n_shots}_shots/'
    }

    def __init__(
        self,
        backbone_name,
        source,
        model_url,
        trainer_name,
        feature_aggregation_config,
        loss_config,
        modalities,
        image_transforms_config,
        image_augmentation_config,
        do_augmentation=False,
        fine_to_coarse_label_id_mapping=None,
        labels_sample_count=None,
    ):
        """
        Initializes a new instance of the ModelConfig class.

        Args:
            backbone_name (str): The name of the backbone model.
            source (str): The source of the model.
            model_url (str): The URL of the model.
            trainer_name (str): The name of the trainer.
            feature_aggregation_config (dict): The configuration for feature aggregation.
            loss_config (dict): The configuration for loss.
            modalities (list): The list of modalities.
            image_transforms_config (dict): The configuration for image transforms.
            image_augmentation_config (dict): The configuration for image augmentation.
            do_augmentation (bool, optional): Whether to perform data augmentation. Defaults to False.
            fine_to_coarse_label_id_mapping (dict, optional): The mapping from fine to coarse label IDs. Defaults to None.
            labels_sample_count (int, optional): The count of labels to sample. Defaults to None.
        """
        self.backbone_name = backbone_name
        self.source = source
        self.model_url = model_url
        self.trainer_name = trainer_name
        self.feature_aggregation_config = feature_aggregation_config
        self.loss_config = loss_config
        self.modalities = modalities
        self.image_transforms_config = image_transforms_config
        self.image_augmentation_config = image_augmentation_config
        self.do_augmentation = do_augmentation
        self.labels = None
        self.label_id_prompt_id_mapping = None
        self.soft_prompting_config = None
        self.soft_prompting_zs_config = None
        self.fine_to_coarse_label_id_mapping = fine_to_coarse_label_id_mapping
        self.labels_sample_count = labels_sample_count

    @classmethod
    def get_default_save_path(cls, setup):
        rel_path = setup.get_relative_save_path()
        return EXPERIMENTS_MODEL_DIR + rel_path + 'pytorch_model.bin'

    @classmethod
    def get_default_load_path(cls, setup):
        rel_path = setup.get_pretrained_checkpoint_relative_path()
        return EXPERIMENTS_MODEL_DIR + rel_path + 'pytorch_model.bin'

    def get_labels_sample_count(self):
        return self.labels_sample_count

    def set_labels_sample_count(self, value):
        self.labels_sample_count = value

    def get_fine_to_coarse_label_id_mapping(self):
        return self.fine_to_coarse_label_id_mapping

    def set_fine_to_coarse_label_id_mapping(self, value):
        self.fine_to_coarse_label_id_mapping = value

    def get_backbone_name(self):
        return self.backbone_name

    def set_backbone_name(self, value):
        self.backbone_name = value

    def get_source(self):
        return self.source

    def set_source(self, value):
        self.source = value

    def get_model_url(self):
        return self.model_url

    def set_model_url(self, value):
        self.model_url = value

    def get_trainer_name(self):
        return self.trainer_name

    def set_do_augmentation(self, value):
        self.do_augmentation= value

    def get_do_augmentation(self):
        return self.do_augmentation

    def set_trainer_name(self, value):
        self.trainer_name = value

    def get_feature_aggregation_config(self):
        return self.feature_aggregation_config

    def set_feature_aggregation_config(self, value):
        self.feature_aggregation_config = value

    def get_loss_config(self, **kwargs):
        return self.loss_config

    def set_loss_config(self, value):
        self.loss_config = value

    def get_modalities(self):
        return self.modalities

    def set_modalities(self, value):
        self.modalities = value

    def get_image_transforms_config(self):
        return self.image_transforms_config

    def set_image_transforms_config(self, value):
        self.image_transforms_config = value

    def get_image_augmentation_config(self):
        return self.image_augmentation_config

    def set_image_augmentation_config(self, value):
        self.image_augmentation_config = value

    def get_labels(self):
        return self.labels

    def set_labels(self, value):
        self.labels = value

    def get_label_id_prompt_id_mapping(self):
        return self.label_id_prompt_id_mapping

    def set_label_id_prompt_id_mapping(self, value):
        self.label_id_prompt_id_mapping = value

    def get_soft_prompting_config(self):
        return self.soft_prompting_config

    def set_soft_prompting_config(self, value):
        self.soft_prompting_config = value

    def get_soft_prompting_zs_config(self):
        return self.soft_prompting_zs_config

    def set_soft_prompting_zs_config(self, value):
        self.soft_prompting_zs_config = value

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        attributes = [
            f"{attr}={getattr(self, attr)!r}" for attr in [
                'backbone_name', 'source', 'model_url', 'trainer_name',
                'feature_aggregation_config', 'loss_config', 'modalities',
                'image_transforms_config', 'labels', 'label_id_prompt_id_mapping',
                'soft_prompting_config', 'soft_prompting_zs_config', 'do_augmentation', 'fine_to_coarse_label_id_mapping',
            ] if getattr(self, attr, None) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(attributes)})"

    
class FeatureAggregationConfig(BaseConfig):
    config = {
        'aggregate_after': {
            'embedding': {
                'enabled': True,
            },
            'loss': {
                'enabled': False,
            },
        },
        'scatter_strategy': 'mean'
    }

    @staticmethod
    def get_config(**kwargs):
        return FeatureAggregationConfig.config


class TrainersAvailabilityConfig(BaseConfig):
    config = {
        Trainers.CLIP: [Sources.OPEN_AI, Sources.HUGGINGFACE],
        Trainers.COOP: [Sources.OPEN_AI],
    }

    @staticmethod
    def get_config(trainer_name, source=None):
        trainer_config = TrainersAvailabilityConfig.get(trainer_name)
        if not source:
            return trainer_config
            
        if source not in trainer_config:
            raise f'Model has not been implemented for {source}'

        return trainer_config

class ModelConfigManager:
    @staticmethod
    def get_config(
        backbone_name=CLIPBackbones.CLIP_VIT_B_32,
        source=Sources.HUGGINGFACE,
        trainer_name=Trainers.CLIP,
        do_augmentation=False,
        fine_to_coarse_label_id_mapping=None,
        labels_sample_count=None,
        loss_type=LossType.CONTRASTIVE_LOSS,
        loss_kwargs={},
        **kwargs,
    ):
        assert source in TrainersAvailabilityConfig.get(trainer_name), f'Model has not been implemented for {source}'
        url = BackboneURLConfig.get_config(backbone_name=backbone_name, source=source)
        modalities = TrainersModalityConfig.get_config(trainer_name, source)
        agg_config = FeatureAggregationConfig.get_config()


        if loss_type is not None:
            loss_config_per_stage = LossConfig.get_config(loss_type=loss_type, **loss_kwargs)
            loss_config = TrainerLoss({stage: loss_config_per_stage for stage in Stages.get_values()})
        else:
            loss_config = TrainersLossConfig.get_config(trainer_name, **loss_kwargs)

        image_transforms_config = TransformationConfigManager.get_transformation_args(backbone_name)
        if do_augmentation:
            image_augmentation_config = TransformationConfigManager.get_transformation_args(backbone_name, trainer_name=trainer_name)
        else:
            image_augmentation_config = {}

        model_config = ModelConfig(
            backbone_name=backbone_name,
            source=source,
            model_url=url,
            trainer_name=trainer_name,
            feature_aggregation_config=agg_config,
            loss_config=loss_config,
            modalities=modalities,
            image_transforms_config=image_transforms_config,
            image_augmentation_config=image_augmentation_config,
            do_augmentation=do_augmentation,
            fine_to_coarse_label_id_mapping=fine_to_coarse_label_id_mapping,
            labels_sample_count=labels_sample_count,
        )

        if trainer_name in SoftPromptingConfig.config:
            labels = kwargs['labels']
            label_id_prompt_id_mapping = kwargs['label_id_prompt_id_mapping']
            context_initialization = kwargs.get('context_initialization', None)

            soft_prompting_config = SoftPromptingConfig.get_config(trainer_name)
            soft_prompting_zs_config = SoftPromptingConfig.get_zero_shot_config()
            if context_initialization:
                soft_prompting_config.set_text_context_initialization(context_initialization)
    
            model_config.set_labels(labels)
            model_config.set_label_id_prompt_id_mapping(label_id_prompt_id_mapping)
            model_config.set_soft_prompting_config(soft_prompting_config)
            model_config.set_soft_prompting_zs_config(soft_prompting_zs_config)
    
        return model_config
