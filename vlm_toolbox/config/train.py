import torch

from config.base import BaseConfig
from config.enums import (
    CLIPBackbones,
    PrecisionDtypes,
    Setups,
    Stages,
    Trainers,
)
from config.path import EXPERIMENTS_LOGGING_DIR


class DefaultLearningRateConfig(BaseConfig):
    config = {
        CLIPBackbones.CLIP_RESNET_50: 0.0005,
        CLIPBackbones.CLIP_RESNET_101: 0.0005,
        CLIPBackbones.CLIP_RESNET_50_4: 0.0005,
        CLIPBackbones.CLIP_RESNET_50_16: 0.0004,
        CLIPBackbones.CLIP_RESNET_50_64: 0.00036,
        CLIPBackbones.CLIP_VIT_B_32: 0.0004,
        CLIPBackbones.CLIP_VIT_B_16: 0.0004,
        CLIPBackbones.CLIP_VIT_L_14: 0.0004,
        CLIPBackbones.CLIP_VIT_L_14_336PX: 2e-05,
        'DEFAULT': 2e-5,
    }

    @staticmethod
    def get_config(backbone_name):
        return  DefaultLearningRateConfig.get(backbone_name) or DefaultLearningRateConfig.config['DEFAULT']

class TrainersBatchSizeConfig(BaseConfig):
    config = {
        Trainers.CLIP: {
            Stages.TRAIN: 512,
            Stages.EVAL: 1024,
            Stages.PREPROCESS: 1024,
        },
        Trainers.COOP: {
            Stages.TRAIN: 256,
            Stages.EVAL: 1024,
            Stages.PREPROCESS: 1024,
        },
    }
    
    @staticmethod
    def get_config(trainer_name=Trainers.CLIP):
        return TrainersBatchSizeConfig.get(trainer_name)

class TrainersOptimConfig(BaseConfig):
    config = {
        'RESNET': {
            'optimizer': {
                'optim': 'adamw_torch',
                'learning_rate': None,
                'adam_beta1': 0.9,
                'adam_beta2': 0.999,
                'adam_epsilon': 1.0e-8,
                'weight_decay': 0.1,
            },
            'lr_scheduler': {
                'lr_scheduler_type': 'cosine',
                'warmup_ratio': 0.05,
            },
            'early_stopping': {
                'early_stopping_patience': 20,
                'early_stopping_threshold': 0.0001,
            },
        },
        'VIT': {
            Trainers.CLIP: {
                'optimizer': {
                    'optim': 'adamw_torch',
                    'learning_rate': None,
                    'adam_beta1': 0.9,
                    'adam_beta2': 0.999,
                    'adam_epsilon': 1.0e-8,
                    'weight_decay': 0.1,
                },
                'lr_scheduler': {
                    'lr_scheduler_type': 'cosine',
                    'warmup_ratio': 0.05,
                },
                'early_stopping': {
                    'early_stopping_patience': 20,
                    'early_stopping_threshold': 0.0001,
                },
            },
            Trainers.COOP: {
                'optimizer': {
                    'optim': 'sgd',
                    'weight_decay': 0.1,
                    'learning_rate': 0.002,
                },
                'lr_scheduler': {
                    'lr_scheduler_type': 'cosine',
                    'warmup_ratio': 0.05,
                },
                'early_stopping': {
                    'early_stopping_patience': 20,
                    'early_stopping_threshold': 0.0001,
                },
            },
        },
    }

    @staticmethod
    def get_config(backbone_name=CLIPBackbones.CLIP_VIT_B_32, trainer_name=Trainers.CLIP):
        default_lr = DefaultLearningRateConfig.get_config(backbone_name=backbone_name)
        optim_config = {}
        if 'resnet' in backbone_name:
            optim_config = TrainersOptimConfig.config['RESNET']
        else:
            optim_config = optim_config = TrainersOptimConfig.config['VIT'][trainer_name]

        if not optim_config['optimizer']['learning_rate']:
            optim_config['optimizer']['learning_rate'] = default_lr

        return optim_config

class TrainingArgumentsConfig(BaseConfig):
    config = {
        'output_dir': EXPERIMENTS_LOGGING_DIR,
        'remove_unused_columns': False,
        'include_inputs_for_metrics': False,
        'evaluation_strategy': 'epoch',
        'report_to': 'tensorboard',
        'tf32': None,
        'fp16': False,
        'bf16': False,
        'save_safetensors': False,
    }
    
    train_eval_config = {
        'evaluation_strategy': 'epoch',
        'logging_strategy': 'epoch',
        'save_total_limit': 1,
        'save_only_model': True,
        'save_strategy': 'epoch',
        'load_best_model_at_end': True,
        'do_train': True,
        'do_eval': True,
        'fp16_full_eval': False,
        'bf16_full_eval': False,
        'gradient_accumulation_steps': 4,
    }

    train_only_config = {
        'do_train': True,
        'do_eval': False,
        'logging_strategy': 'epoch',
        'gradient_accumulation_steps': 4,
    }
    
    eval_only_config = {
        'do_eval': True,
        'do_train': False,
        'fp16_full_eval': False,
        'bf16_full_eval': False,
    }

    @staticmethod
    def get_config(
        setup_type=Setups.FULL,
        precision_dtype=False,
        train_full_precision=False,
        eval_full_precision=False,
        tf32=None,
        auto_find_batch_size=False,
        train_batch_size=None,
        eval_batch_size=None,
        num_epochs=None,
        metric_for_best_model=None,
        label_names=None,
        greater_is_better=None,
        **kwargs,
    ):
        default_config = TrainingArgumentsConfig.config
        if setup_type == Setups.EVAL_ONLY:
            default_config = {**default_config, **TrainingArgumentsConfig.eval_only_config}
        elif setup_type == Setups.TRAIN_ONLY:
            default_config = {**default_config, **TrainingArgumentsConfig.train_only_config}
        elif setup_type == Setups.FULL:
            default_config = {**default_config, **TrainingArgumentsConfig.train_eval_config}
        else:
            raise ValueError('Invalid setup type.')

        if tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                torch.backends.cudnn.allow_tf32 = tf32
                default_config['tf32'] = True if tf32 else None
            except:
                default_config['tf32'] = None

        if not train_full_precision:
            if precision_dtype == PrecisionDtypes.FP16:
                default_config['fp16'] = True
            elif precision_dtype == PrecisionDtypes.BF16 and torch.cuda.is_bf16_supported():
                default_config['bf16'] = True
        
        if setup_type != Setups.EVAL_ONLY:
            if train_batch_size and isinstance(train_batch_size, int):
                default_config['per_device_train_batch_size'] = train_batch_size
            if num_epochs and isinstance(num_epochs, int):
                default_config['num_train_epochs'] = num_epochs

        if setup_type != Setups.TRAIN_ONLY:
            if eval_batch_size and isinstance(eval_batch_size, int):
                default_config['per_device_eval_batch_size'] = eval_batch_size
            
            if not eval_full_precision:
                if precision_dtype == PrecisionDtypes.FP16:
                    default_config['fp16_full_eval'] = True
                elif precision_dtype == PrecisionDtypes.BF16 and torch.cuda.is_bf16_supported():
                    default_config['bf16_full_eval'] = True
            
            default_config['metric_for_best_model'] = metric_for_best_model
            default_config['label_names'] = label_names
            default_config['greater_is_better'] = greater_is_better

        default_config['auto_find_batch_size'] = auto_find_batch_size
        return default_config
