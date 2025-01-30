import os
from collections import defaultdict
from itertools import product

import dill as pickle

from config.base import BaseConfig
from config.enums import (
    CLIPBackbones,
    ImageDatasets,
    LossType,
    Metrics,
    ModelType,
    PrecisionDtypes,
    SamplingStrategy,
    SamplingType,
    Setups,
    Sources,
    Stages,
    Trainers,
)
from config.logging import log_tree
from config.path import EXPERIMENTS_MODEL_DIR, EXPERIMENTS_ROOT_DIR, SETUPS_DIR
from config.soft_prompting import SoftPromptingConfig
from config.train import TrainersBatchSizeConfig
from config.validator import Validator

RANDOM_STATE = 42

TEMPLATE_DIR_DICT = {
    'model_type': '{model_type}',
    'dataset_name': '{dataset_name}',
    'backbone_name': '{backbone_name}',
    'trainer_name': '{trainer_name}',
    'source': '{source}',
    'main_metric_name': 'metric_{main_metric_name}',
    'loss_type': 'loss_{loss_type}',
    'n_shots': '{n_shots}_shots/',
    'label_column_name': 'column_{label_column_name}',
    'sampling_type': '{sampling_type}',
    'sampling_strategy': 'sampling_strategy_{sampling_strategy}',
}

class TrainerSetupDefaults(BaseConfig):
    config = {
        Trainers.CLIP: {
            'setup_type': Setups.EVAL_ONLY,
            'backbone_name': CLIPBackbones.CLIP_VIT_B_16,
            'source': Sources.OPEN_AI,
            'model_type': ModelType.PRETRAINED,
        },
        Trainers.COOP: {
            'setup_type': Setups.FULL,
            'backbone_name': CLIPBackbones.CLIP_VIT_B_16,
            'source': Sources.OPEN_AI,
            'model_type': ModelType.FEW_SHOT,
        },
    }

    @staticmethod
    def get_config(trainer_name):
        return TrainerSetupDefaults.get(trainer_name)

class Setup:
    save_dir = SETUPS_DIR

    def __init__(
        self, dataset_name, trainer_name, setup_type=None, backbone_name=None, source=None, model_type=None,
        train_full_precision=False, eval_full_precision=False, precision_dtype=PrecisionDtypes.FP16, enable_tensor_float=None, 
        label_column_name=None, annotations_key_value_criteria={}, main_metric_name=Metrics.ACCURACY, num_epochs=200, n_shots=None,
        complementary_metrics_names=[Metrics.F1, Metrics.RECALL, Metrics.PRECISION, Metrics.ACCURACY, Metrics.G_MEAN, Metrics.BALANCED_ACCURACY],
        validation_size=0.15, train_batch_size=None, eval_batch_size=None, preprocess_batch_size=None, 
        train_split=Stages.TRAIN, eval_split=Stages.EVAL, top_k=5, random_state=RANDOM_STATE, model_checkpoint_path=None, 
        auto_batch_size=False, use_dataset_context_init=False, load_from_checkpoint=False, do_augmentation=False, coarse_column_name=None, 
        loss_type=None, loss_kwargs={}, sampling_type=None, sampling_strategy=None, sampling_kwargs={}
    ):
        """
        Initializes an instance of the Setup class.

        Args:
            dataset_name (str): The name of the dataset.
            trainer_name (str): The name of the trainer.
            setup_type (str, optional): The setup type. Defaults to None.
            backbone_name (str, optional): The name of the backbone. Defaults to None.
            source (str, optional): The source of the data. Defaults to None.
            model_type (str, optional): The type of the model. Defaults to None.
            train_full_precision (bool, optional): Whether to train in full precision. Defaults to False.
            eval_full_precision (bool, optional): Whether to evaluate in full precision. Defaults to False.
            precision_dtype (str, optional): The precision data type. Defaults to PrecisionDtypes.FP16.
            enable_tensor_float (bool, optional): Whether to enable tensor float. Defaults to None.
            label_column_name (str, optional): The name of the label column. Defaults to None.
            annotations_key_value_criteria (dict, optional): The key-value criteria for annotations. Defaults to {}.
            main_metric_name (str, optional): The name of the main metric. Defaults to Metrics.ACCURACY.
            complementary_metrics_names (list, optional): The names of the complementary metrics. Defaults to [Metrics.F1, Metrics.RECALL, Metrics.PRECISION, Metrics.ACCURACY, Metrics.G_MEAN, Metrics.BALANCED_ACCURACY].
            num_epochs (int, optional): The number of epochs. Defaults to 200.
            n_shots (int, optional): The number of shots. Defaults to None.
            validation_size (float, optional): The size of the validation set. Defaults to 0.15.
            train_batch_size (int, optional): The batch size for training. Defaults to None.
            eval_batch_size (int, optional): The batch size for evaluation. Defaults to None.
            preprocess_batch_size (int, optional): The batch size for preprocessing. Defaults to None.
            train_split (str, optional): The split for training. Defaults to Stages.TRAIN.
            eval_split (str, optional): The split for evaluation. Defaults to Stages.EVAL.
            top_k (int, optional): The value of k for top-k accuracy. Defaults to 5.
            random_state (int, optional): The random state. Defaults to RANDOM_STATE.
            model_checkpoint_path (str, optional): The path to the model checkpoint. Defaults to None.
            auto_batch_size (bool, optional): Whether to use automatic batch size. Defaults to False.
            use_dataset_context_init (bool, optional): Whether to use dataset context initialization. Defaults to False.
            load_from_checkpoint (bool, optional): Whether to load from a checkpoint. Defaults to False.
            do_augmentation (bool, optional): Whether to perform data augmentation. Defaults to False.
            coarse_column_name (str, optional): The name of the coarse column. Defaults to None.
            loss_type (str, optional): The type of loss. Defaults to None.
            loss_kwargs (dict, optional): The keyword arguments for the loss. Defaults to {}.
            sampling_type (str, optional): The type of sampling. Defaults to None.
            sampling_strategy (str, optional): The sampling strategy. Defaults to None.
            sampling_kwargs (dict, optional): The keyword arguments for sampling. Defaults to {}.
        """
        self.set_trainer_name(trainer_name)
        self.set_dataset_name(dataset_name)

        defaults = TrainerSetupDefaults.get_config(trainer_name=trainer_name)
        
        self.set_setup_type(setup_type or defaults['setup_type'])
        self.set_backbone_name(backbone_name or defaults['backbone_name'])
        self.set_source(source or defaults['source'])
        self.set_model_type(model_type or defaults['model_type'])
        self.is_soft = SoftPromptingConfig.is_valid(self.trainer_name, strict=True)

        self.set_n_shots(n_shots)
        self.set_train_full_precision(train_full_precision)
        self.set_eval_full_precision(eval_full_precision)
        self.set_precision_dtype(precision_dtype)
        self.set_enable_tensor_float(enable_tensor_float)
        self.set_label_column_name(label_column_name)
        self.set_annotations_key_value_criteria(annotations_key_value_criteria)
        self.set_main_metric_name(main_metric_name)
        self.set_complementary_metrics_names(complementary_metrics_names)
        self.set_num_epochs(num_epochs)
        self.set_validation_size(validation_size)
        self.default_batch_sizes = TrainersBatchSizeConfig.get_config(trainer_name=trainer_name)
        self.set_train_batch_size(train_batch_size)
        self.set_eval_batch_size(eval_batch_size)
        self.set_preprocess_batch_size(preprocess_batch_size)
        self.set_train_split(train_split)
        self.set_eval_split(eval_split)
        self.set_top_k(top_k)
        self.set_random_state(random_state)
        self.set_model_checkpoint_path(model_checkpoint_path)
        self.set_auto_batch_size(auto_batch_size)
        self.set_use_dataset_context_init(use_dataset_context_init)
        self.set_load_from_checkpoint(load_from_checkpoint)
        self.set_do_augmentation(do_augmentation)
        self.set_coarse_column_name(coarse_column_name)
        self.set_loss_type(loss_type)
        self.set_loss_kwargs(loss_kwargs)
        self.set_sampling_type(sampling_type)
        self.set_sampling_strategy(sampling_strategy)
        self.set_sampling_kwargs(sampling_kwargs)
        self.id = None

    def get_pretrained_checkpoint_relative_path(self):
        return self._get_relative_path(exclude_keys=['loss_type', 'label_column_name'])

    def get_relative_save_path(self):
        path = self._get_relative_path()
        if self.annotations_key_value_criteria:
            parts = [f"{key}__{'_'.join(values)}" for key, values in self.annotations_key_value_criteria.items()]
            path = os.path.join(path, 'filters_' + '__'.join(parts))
        return self._ensure_trailing_slash(path)

    def _get_relative_path(self, exclude_keys=[]):
        template_dir_dict = {k: v for k, v in TEMPLATE_DIR_DICT.items() if k not in exclude_keys and getattr(self, k) is not None}
        path = os.path.join(*template_dir_dict.values()).format(**self.to_dict())
        return self._ensure_trailing_slash(path)

    def _ensure_trailing_slash(self, path):
        return path if path.endswith('/') else path + '/'

    def set_trainer_name(self, value):
        Validator.validate_enum_value(value, Trainers, "trainer_name")
        self.trainer_name = value

    def set_dataset_name(self, value):
        Validator.validate_enum_value(value, ImageDatasets, "dataset_name")
        self.dataset_name = value

    def set_setup_type(self, value):
        Validator.validate_enum_value(value, Setups, "setup_type")
        self.setup_type = value

    def set_backbone_name(self, value):
        Validator.validate_enum_value(value, CLIPBackbones, "backbone_name")
        self.backbone_name = value

    def set_source(self, value):
        Validator.validate_enum_value(value, Sources, "source")
        self.source = value

    def set_model_type(self, value):
        Validator.validate_enum_value(value, ModelType, "model_type")
        self.model_type = value

    def set_n_shots(self, value):
        if self.setup_type != Setups.EVAL_ONLY or self.is_soft:
            Validator.validate_non_negative_int(value, "n_shots")
            self.n_shots = value
        else:
            self.n_shots = None

    def set_validation_size(self, value):
        if self.setup_type != Setups.EVAL_ONLY:
            Validator.validate_positive_number(value, "validation_size")
            self.validation_size = value
        else:
            self.validation_size = None

    def set_num_epochs(self, value):
        if self.setup_type != Setups.EVAL_ONLY:
            Validator.validate_positive_int(value, "num_epochs")
            self.num_epochs = value
        else:
            self.num_epochs = None

    def set_train_batch_size(self, value):
        if self.setup_type != Setups.EVAL_ONLY:
            self.train_batch_size = value or self.default_batch_sizes[Stages.TRAIN]
        else:
            self.train_batch_size = None

    def set_eval_batch_size(self, value):
        if self.setup_type != Setups.TRAIN_ONLY:
            self.eval_batch_size = value or self.default_batch_sizes[Stages.EVAL]
        else:
            self.eval_batch_size = None

    def set_preprocess_batch_size(self, value):
        self.preprocess_batch_size = value or self.default_batch_sizes[Stages.PREPROCESS]

    def set_train_full_precision(self, value):
        self.train_full_precision = self._validate_flag(value, "train_full_precision", Setups.EVAL_ONLY)

    def set_eval_full_precision(self, value):
        self.eval_full_precision = self._validate_flag(value, "eval_full_precision", Setups.TRAIN_ONLY, default=False)

    def set_enable_tensor_float(self, value):
        if value is None or isinstance(value, bool):
            self.enable_tensor_float = value
        else:
            raise ValueError(f"Invalid enable tensor float: {value}")

    def _validate_flag(self, value, flag_name, invalid_setup, default=None):
        if self.setup_type == invalid_setup:
            return default
        if isinstance(value, bool):
            return value
        raise ValueError(f"Invalid {flag_name} flag: {value}")

    def set_precision_dtype(self, value):
        Validator.validate_enum_value(value, PrecisionDtypes, "precision_dtype")
        self.precision_dtype = value

    def set_label_column_name(self, value):
        Validator.validate_optional_string(value, "label_column_name")
        self.label_column_name = value

    def set_coarse_column_name(self, value):
        Validator.validate_optional_string(value, "coarse_column_name")
        self.coarse_column_name = value

    def set_annotations_key_value_criteria(self, value):
        Validator.validate_dict(value, "annotations_key_value_criteria")
        self.annotations_key_value_criteria = value

    def set_main_metric_name(self, value):
        Validator.validate_enum_value(value, Metrics, "main_metric_name")
        self.main_metric_name = value

    def set_complementary_metrics_names(self, value):
        Validator.validate_list_of_enum_values(value, Metrics, "complementary_metrics_names")
        self.complementary_metrics_names = value

    def set_top_k(self, value):
        Validator.validate_any(
            [Validator.validate_positive_int, Validator.validate_inf],
            value, "top_k",
        )
        self.top_k = value

    def set_random_state(self, value):
        Validator.validate_positive_int(value, "random_state")
        self.random_state = value

    def set_model_checkpoint_path(self, value):
        if value is None or (isinstance(value, str) and os.path.isdir(os.path.dirname(value))):
            self.model_checkpoint_path = value
        else:
            raise ValueError(f"Invalid model_checkpoint_path: {value}")

    def set_auto_batch_size(self, value):
        Validator.validate_bool(value, "auto_batch_size")
        self.auto_batch_size = value

    def set_use_dataset_context_init(self, value):
        Validator.validate_bool(value, "use_dataset_context_init")
        self.use_dataset_context_init = value

    def set_load_from_checkpoint(self, value):
        Validator.validate_bool(value, "load_from_checkpoint")
        self.load_from_checkpoint = value

    def set_do_augmentation(self, value):
        Validator.validate_bool(value, "do_augmentation")
        self.do_augmentation = value

    def set_loss_type(self, value):
        if value is None or value in LossType.get_values():
            self.loss_type = None if value == LossType.CONTRASTIVE_LOSS else value
        else:
            raise ValueError(f"Invalid loss_type: {value}")

    def set_loss_kwargs(self, value):
        Validator.validate_dict(value, "loss_kwargs")
        self.loss_kwargs = value

    def set_train_split(self, value):
        Validator.validate_enum_value(value, Stages, "train_split")
        self.train_split = value
    
    def set_eval_split(self, value):
        Validator.validate_enum_value(value, Stages, "eval_split")
        self.eval_split = value
    
    def set_load_from_checkpoint(self, value):
        Validator.validate_bool(value, "load_from_checkpoint")
        self.load_from_checkpoint = value

    def set_sampling_type(self, value):
        if value is None:
            self.sampling_type = None
        else:
            Validator.validate_enum_value(value, SamplingType, "sampling_type")
            self.sampling_type = value

    def set_sampling_strategy(self, value):
        if value is None:
            self.sampling_strategy = None
        else:
            Validator.validate_enum_value(value, SamplingStrategy, "sampling_strategy")
            self.sampling_strategy = value

    def set_sampling_kwargs(self, value):
        Validator.validate_dict(value, "sampling_kwargs")
        self.sampling_kwargs = value

    def get_enable_tensor_float(self):
        return self.enable_tensor_float

    def get_do_augmentation(self):
        return self.do_augmentation

    def get_load_from_checkpoint(self):
        return self.load_from_checkpoint or (self.get_model_type() == ModelType.PRETRAINED and self.get_trainer_name() != Trainers.CLIP)

    def get_use_dataset_context_init(self):
        return self.use_dataset_context_init

    def get_auto_batch_size(self):
        return self.auto_batch_size

    def get_model_checkpoint_path(self):
        if self.model_checkpoint_path is not None:
            return self.model_checkpoint_path
        elif self.get_load_from_checkpoint:
            return EXPERIMENTS_MODEL_DIR + self.get_relative_save_path() + 'pytorch_model.bin'
        return None

    def get_setup_type(self):
        return self.setup_type

    def get_dataset_name(self):
        return self.dataset_name

    def get_backbone_name(self):
        return self.backbone_name

    def get_source(self):
        return self.source

    def get_trainer_name(self):
        return self.trainer_name

    def get_model_type(self):
        return self.model_type

    def get_label_column_name(self):
        return self.label_column_name

    def get_coarse_column_name(self):
        return self.coarse_column_name

    def get_annotations_key_value_criteria(self):
        return self.annotations_key_value_criteria

    def get_main_metric_name(self):
        return self.main_metric_name

    def get_completion_metrics_names(self):
        return self.complementary_metrics_names

    def get_num_epochs(self):
        return self.num_epochs

    def get_n_shots(self):
        return self.n_shots

    def get_validation_size(self):
        return self.validation_size

    def get_train_batch_size(self):
        return self.train_batch_size

    def get_eval_batch_size(self):
        return self.eval_batch_size

    def get_preprocess_batch_size(self):
        return self.preprocess_batch_size

    def get_train_full_precision(self):
        return self.train_full_precision

    def get_eval_full_precision(self):
        return self.eval_full_precision

    def get_precision_dtype(self):
        return self.precision_dtype
    
    def get_top_k(self):
        return self.top_k

    def get_random_state(self):
        return self.random_state

    def get_is_soft(self):
        return self.is_soft

    def get_loss_type(self):
        return self.loss_type
    
    def get_loss_kwargs(self):
        return self.loss_kwargs

    def get_sampling_type(self):
        return self.sampling_type

    def get_sampling_strategy(self):
        return self.sampling_strategy

    def get_sampling_kwargs(self):
        return self.sampling_kwargs

    def get_split(self, stage):
        if stage == Stages.TRAIN:
            return self.train_split
        elif stage == Stages.EVAL:
            return self.eval_split
        else:
            raise ValueError(f'Unsupported stage: {stage}')

    def get_stages(self):
        setup_type = self.get_setup_type()
        if setup_type == Setups.FULL:
            return [Stages.TRAIN, Stages.EVAL]
        if setup_type == Setups.TRAIN_ONLY:
            return [Stages.TRAIN]
        if setup_type == Setups.EVAL_ONLY:
            return [Stages.EVAL]
        else:
            raise ValueError('Invalid setup type')

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({attrs})"

    def save(self):
        directory = self.save_dir
        os.makedirs(directory, exist_ok=True)
        existing_files = [int(f.split('.')[0].replace('setup_', '')) for f in os.listdir(directory) if f.endswith('.pkl')]
        next_file_number = max(existing_files, default=0) + 1
        file_path = os.path.join(directory, f"setup_{next_file_number}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        self.id = next_file_number
        return file_path

    @staticmethod
    def load(file_number):
        file_path = os.path.join(Setup.save_dir, f"setup_{file_number}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found with the number {file_number}")
        with open(file_path, 'rb') as file:
            setup = pickle.load(file)
        setup.id = file_number
        return setup

    def delete(self):
        if self.id:
            os.remove(os.path.join(Setup.save_dir, f"setup_{self.id}.pkl"))
            self.id = None

    @classmethod
    def list_available_experiment_results(cls, display=True, with_fnames=False, with_dirs=False,  **kwargs):
        return cls._list_available_files(
            EXPERIMENTS_ROOT_DIR,
            file_validator=lambda f: f.endswith(('.csv', '.parquet')),
            display=display,
            with_fnames=with_fnames,
            with_dirs=with_dirs,
            **kwargs,
        )

    @classmethod
    def list_available_models(cls, display=True, with_fnames=False, with_dirs=False, **kwargs):
        return cls._list_available_files(
            EXPERIMENTS_MODEL_DIR,
            file_validator=lambda f: f == 'pytorch_model.bin',
            display=display,
            with_fnames=with_fnames,
            with_dirs=with_dirs,
            **kwargs,
        )

    @classmethod
    def _list_available_files(
        cls, base_dir,
        file_validator,
        display=True,
        with_fnames=False,
        with_dirs=False, 
        **kwargs,
    ):
        results, dirs_dict = cls._get_available_files(base_dir, file_validator, with_fnames=with_fnames, **kwargs)
        if display:
            if with_dirs:
                log_tree(dirs_dict)
            else:
                log_tree(results)
        else:
            return dirs_dict if with_dirs else results

    @classmethod
    def _get_available_files(cls, base_dir, file_validator, **kwargs):
        def _nested_defaultdict(depth, default_type=set):
            if depth == 1:
                return defaultdict(default_type)
            return defaultdict(lambda: _nested_defaultdict(depth - 1, default_type))

        def _convert_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: _convert_to_dict(v) for k, v in d.items()}
            return list(d) if isinstance(d, set) else d

        def _parse_filter_part(filter_part):
            if not filter_part:
                return {}
            filter_criteria = filter_part[len('filters_'):].split('__')
            return {filter_criteria[0]: filter_criteria[1].split('_')}

        def _extract_attrs_from_path(root):
            parts = root.split('/')
            label_column_name = next((part[len('column_'):] for part in parts if part.startswith('column_')), None)
            loss_type = next((part[len('loss_'):] for part in parts if part.startswith('loss_')), None)
            n_shots = next((part.split('_')[0] for part in parts if part.endswith('_shots')), None)
            sampling_type = next((part.split('_')[0] for part in parts if part.endswith('_sampling')), None)
            sampling_strategy = next((part[len('sampling_strategy_'):] for part in parts if part.startswith('sampling_strategy_')), None)

            filter_part = next((part for part in parts if part.startswith('filters_')), None)
            filter_dict = _parse_filter_part(filter_part)

            return n_shots, label_column_name, loss_type, filter_dict, sampling_type, sampling_strategy
    
        def _file_str_formatter(n_shots, label_column_name, loss_type, filter_dict, sampling_type, sampling_strategy):
            label_column_str = f"label: {label_column_name}" if label_column_name else "label: default"
            loss_str = f"{loss_type} loss" if loss_type else "contrastive loss"
            shots_str = f"{n_shots} shots" if n_shots else "no shots"
            sampling_type_str = f"{sampling_type}" if sampling_type else "no sampling"
            if sampling_type:
                sampling_strategy_str = f"sampling strategy: {sampling_strategy}" if sampling_strategy else "no sampling strategy"
            else:
                sampling_strategy_str = ""

            filter_str = "filters: {}"
            if filter_dict:
                filter_str = f"filters: {', '.join([f'feature {k} ∈ {v}' for k, v in filter_dict.items()])}"

            return shots_str, loss_str, label_column_str, filter_str, sampling_type_str, sampling_strategy_str

        def _ensure_list(value):
            return value if isinstance(value, list) else [value]

        def _prune_dict(d):
            def _simplify_dict(d, parent_key='', depth=0):
                if not isinstance(d, dict):
                        return d
                new_dict = {}
                for key, value in d.items():
                    if isinstance(value, dict):
                        if len(value) == 1 and depth > 2:
                            sub_key = next(iter(value))
                            sub_key_cleaned = (
                                sub_key.replace('no shots', '').strip().strip('-').strip()
                               .replace('label: default', '').strip().strip('-').strip()
                               .replace('filters: {}', '').strip().strip('-').strip()
                               .replace('no sampling strategy', '').strip().strip('-').strip()
                               .replace('no sampling', '').strip().strip('-').strip()
                            )
                            new_key = f"{key} - {sub_key_cleaned}".strip().strip('-').strip()
                            new_dict[new_key] = _simplify_dict(value[sub_key], new_key, depth + 1)
                        else:
                            new_dict[key] = _simplify_dict(value, key, depth + 1)
                    else:
                        new_dict[key] = value
                return new_dict
            return _simplify_dict(_simplify_dict(d))

        with_file_names = kwargs.pop('with_fnames', False)
        depth = 10 if with_file_names else 9
        dirs_dict = _nested_defaultdict(depth)
        results = _nested_defaultdict(depth)
        all_roots = set()

        for mt, tn, bn, src, dn, lt in product(*map(_ensure_list, [
            kwargs.get('model_type', ModelType.get_values()),
            kwargs.get('trainer_name', Trainers.get_values()),
            kwargs.get('backbone_name', CLIPBackbones.get_values()),
            kwargs.get('source', Sources.get_values()),
            kwargs.get('dataset_name', ImageDatasets.get_values()),
            kwargs.get('loss_type', LossType.get_values()),
        ])):
            setup = cls(
                dataset_name=dn,
                trainer_name=tn,
                backbone_name=bn,
                model_type=mt,
                source=src,
                loss_type=lt,
                **kwargs,
            )
            rel_path = setup.get_pretrained_checkpoint_relative_path() if mt == ModelType.PRETRAINED else setup.get_relative_save_path()
            full_path = os.path.join(base_dir, rel_path)
            if os.path.exists(full_path):
                for root, _, files in os.walk(full_path):
                    for file in files:
                        if file_validator(file):
                            absolute_dir = os.path.join(root, file)
                            n_shots, label_column_name, loss_type, filter_dict, sampling_type, sampling_strategy = _extract_attrs_from_path(root)
                            if mt != ModelType.PRETRAINED:
                                n_shots = n_shots or 16
                            shots_str, loss_str, label_column_str, filter_str, sampling_type_str, sampling_strategy_str = _file_str_formatter(
                                n_shots, label_column_name, loss_type, filter_dict, sampling_type, sampling_strategy
                            )
                            model_str = f"{label_column_str}".strip().strip('-').strip()

                            if root not in all_roots:
                                all_roots.add(root)
                                if with_file_names:
                                    results[mt][tn][bn][dn][sampling_type_str][sampling_strategy_str][loss_str][shots_str][filter_str][model_str].add(file)
                                    dirs_dict[mt][tn][bn][dn][sampling_type_str][sampling_strategy_str][loss_str][shots_str][filter_str][model_str].add(absolute_dir)
                                else:
                                    results[mt][tn][bn][dn][sampling_type_str][sampling_strategy_str][loss_str][shots_str][filter_str].add(model_str)
                                    dirs_dict[mt][tn][bn][dn][sampling_type_str][sampling_strategy_str][loss_str][shots_str][filter_str].add(absolute_dir)

        pruned_results = _prune_dict(_convert_to_dict(results))
        pruned_dirs_dict = _prune_dict(_convert_to_dict(dirs_dict))
        return pruned_results, pruned_dirs_dict
