from config.base import BaseConfig
from config.enums import CLIPBackbones, Trainers


class TransformationConfigManager:
    @staticmethod
    def get_transformation_args(backbone_name, trainer_name=None):     
        backbone_config = BackbonesTransformsConfig.get_config(backbone_name)
        trainer_config = TrainersTransformsConfig.get_config(trainer_name) if trainer_name else {}

        return {
            **backbone_config,
            **trainer_config,
        }
        

class TrainersTransformsConfig(BaseConfig):
    config = {
        Trainers.CLIP: {},
        Trainers.COOP: {
            # 'is_training': True,
            # 'train_crop_mode': 'rrc',
            # 'auto_augment': 'rand-m9-mstd0.5-inc1',
            # 're_prob': 0.25,
            # 're_mode': 'pixel',
            # 're_count': 1,
            # 'color_jitter': 0.4,
        },
    }

    @staticmethod
    def get_config(trainer_name=Trainers.CLIP):
        return TrainersTransformsConfig.get(trainer_name)


class BackbonesTransformsConfig(BaseConfig):
    config = {
        CLIPBackbones.CLIP_VIT_B_32: {
            'input_size': (3, 224, 224),
            'interpolation': 'bicubic',
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
        },
        CLIPBackbones.CLIP_VIT_B_16: {
            'input_size': (3, 224, 224),
            'interpolation': 'bicubic',
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
        },
    }

    @staticmethod
    def get_config(backbone_name=CLIPBackbones.CLIP_VIT_B_32):
        model_config = BackbonesTransformsConfig.get(backbone_name)
        return model_config
