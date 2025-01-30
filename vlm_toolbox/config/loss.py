from config.base import BaseConfig
from config.enums import LossType, LossWrappers, Stages, Trainers


DEFAULT_LOSS_CONFIG = {
    'loss_class': LossType.CONTRASTIVE_LOSS,
    'wrapper_class': None,
    'kwargs': {'is_symmetric': False, 'reduction': 'mean'},
}

class LossConfig:
    config = {
        LossType.CONTRASTIVE_LOSS: {
            'is_symmetric': False,
            'reduction': 'mean',
        },
        LossType.LABEL_SMOOTHING_LOSS: {
            'smoothing': 0.1,
            'is_symmetric': False,
            'reduction': 'mean',
        },
        LossType.MARGIN_METRIC_LOSS: {
            'adaptive_weight': 0.2,
            'reduction': 'mean',
        },
        LossType.WEIGHTED_L2_LOSS: {
            'weight': 1.0,
            'reduction': 'mean',
        },
        LossType.WEIGHTED_L1_LOSS: {
            'weight': 1.0,
            'reduction': 'mean',
        },
        LossType.ENLARGED_LARGE_MARGIN_LOSS: {
            'max_m': 0.5,
            'lamda': 1.0,
            'weight': None,
            's': 30,
            'reduction': 'mean',
        },
    }

    wrappers = {
        LossWrappers.COARSELY_SUPERVISED_LOSS: {
            'weight': 0.1,
        }
    }

    loss_wrappers = {
        LossType.CONTRASTIVE_LOSS: None,
        LossType.LABEL_SMOOTHING_LOSS: None,
        LossType.MARGIN_METRIC_LOSS: None,
        LossType.WEIGHTED_L2_LOSS: None,
        LossType.WEIGHTED_L1_LOSS: None
    }

    @staticmethod
    def get_config(loss_type, wrapper_class=None, **kwargs):
        loss_config = LossConfig.config[loss_type].copy()
        loss_config.update(kwargs)
        wrapper_class = LossConfig.loss_wrappers.get(loss_type, None) if not wrapper_class else wrapper_class
        if wrapper_class:
            loss_config.update(LossConfig.wrappers.get(wrapper_class, {}))
        
        return {
            'loss_class': loss_type,
            'wrapper_class': wrapper_class,
            'kwargs': loss_config,
        }

class TrainerLoss:
    def __init__(self, loss_config):
        self.loss_config = loss_config
        self.default_config = DEFAULT_LOSS_CONFIG

        if not isinstance(self.loss_config, dict):
            self.loss_config = {Stages.TRAIN: self.loss_config}

    def get_loss_config(self, stage=Stages.TRAIN):
        if stage in self.loss_config:
            return self.loss_config[stage]
        return self.default_config

    def __repr__(self):
        return f'TrainerLossConfig(loss_config={self.loss_config}, default_config={self.default_config})'

    def from_config(self, *args):
        config = self.loss_config
        if isinstance(args[0], str):
            config = TrainersLossConfig.get_config(args[0]).loss_config
        elif isinstance(args[0], dict):
            config = args[0]
        else:
            raise ValueError("Invalid argument type for from_config")
        self.loss_config = config


class TrainersLossConfig(BaseConfig):
    config = {
        Trainers.CLIP: {
            Stages.TRAIN: LossConfig.get_config(LossType.CONTRASTIVE_LOSS),
            Stages.EVAL: LossConfig.get_config(LossType.CONTRASTIVE_LOSS),
        },
        Trainers.COOP: {
            Stages.TRAIN: LossConfig.get_config(LossType.CONTRASTIVE_LOSS),
            Stages.EVAL: LossConfig.get_config(LossType.CONTRASTIVE_LOSS),
        },
    }

    @staticmethod
    def get_config(trainer_name, **kwargs):
        trainer_config = TrainersLossConfig.config.get(trainer_name)
        if not trainer_config:
            raise ValueError(f"No configuration found for trainer: {trainer_name}")

        stage_config = trainer_config.get(Stages.TRAIN, None)
        if not stage_config:
            raise ValueError(f"No stage configuration found for trainer: {trainer_name}")

        loss_class = stage_config['loss_class']
        wrapper_class = stage_config['wrapper_class']
        default_kwargs = LossConfig.get_config(loss_class, **kwargs)['kwargs']
        default_kwargs.update(stage_config['kwargs'])

        loss_config = {
            Stages.TRAIN: {
                'loss_class': loss_class,
                'wrapper_class': wrapper_class,
                'kwargs': default_kwargs,
            },
        }
        return TrainerLoss(loss_config)
