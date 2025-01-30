import gc
import os
import signal

import psutil
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
import textwrap3

from config.annotations import AnnotationsConfig
from config.enums import (
    DataStatus,
    Modalities,
    ModalityType,
    ModelType,
    Setups,
    Stages,
    Trainers,
)
from config.image_datasets import ImageDatasetConfig
from config.logging import LoggerFactory
from config.modality import ModalityManager
from config.model import ModelConfigManager
from config.precision import PrecisionConfig
from config.train import TrainersOptimConfig, TrainingArgumentsConfig
from data.data_access.image_factory import ImageHandlerFactory
from data.data_access.label_factory import LabelHandleFactory
from data.data_access.text_factory import TextHandlerFactory
from data.data_collate.factory import DataCollatorFactory
from data.data_collate.multimodal_collator import MultiModalDataCollator
from metric.classification import ClassificationMetricEvaluator
from model.vlm_factory import VLMFactory
from util.tensorboard import TensorboardConnector


class Pipeline:
    def __init__(self, setup, device_type='cpu', logger=LoggerFactory.create_logger(name='Pipeline')):
        """
        Initializes a Pipeline object.

        Args:
            setup (Setup): The setup object containing the configuration for the pipeline.
            device_type (str, optional): The type of device to use for training. Defaults to 'cpu'.
            logger (Logger, optional): The logger object to use for logging. Defaults to a new logger instance.

        Attributes:
            setup (Setup): The setup object containing the configuration for the pipeline.
            device_type (str): The type of device to use for training.
            device (torch.device): The device object representing the chosen device.
            logger (Logger): The logger object to use for logging.
            model (torch.nn.Module or None): The model object used for training and inference. Defaults to None.
            trainer (torch.optim.Optimizer or None): The trainer object used for training. Defaults to None.
            stages (List[str]): The list of stages to be executed in the pipeline.
            precision_dtype (PrecisionConfig): The precision configuration object.
            stage_handlers (Dict[int, Dict]): The dictionary containing the stage handlers for each modality.
            stage_handlers_configs (Dict[int, Dict]): The dictionary containing the stage handlers configurations for each modality.
            tensorboard_connector (TensorboardConnector or None): The TensorboardConnector object used for logging. Defaults to None.

        Returns:
            None
        """
        self.setup = setup
        self.device_type = 'cuda' if torch.cuda.is_available() and device_type == 'cuda' else 'cpu'
        self.device = torch.device(self.device_type)
        self.logger = logger

        self.model = None
        self.trainer = None
        self.stages = setup.get_stages()
        self.precision_dtype = PrecisionConfig.get_config(precision_dtype=setup.get_precision_dtype())
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        ModalityManager.flush()
        self.stage_handlers = {modality_idx: {} for modality_idx in Modalities.get_values()}
        self.stage_handlers_configs = {modality_idx: {} for modality_idx in Modalities.get_values()}

        if self.setup.get_setup_type() != Setups.EVAL_ONLY:
            self.tensorboard_connector = TensorboardConnector().start(logging_fn=self.logger.info)

        self._initialize_configs()

    def run(
        self,
        collate_all_m2_samples=False,
        save_predictions=True,
        persist=True,
        **kwargs,
    ):
        self.setup_data()
        self.setup_model()

        if self.setup.get_setup_type() != Setups.EVAL_ONLY:
            self.train(collate_all_m2_samples=collate_all_m2_samples, persist=persist)

        if self.setup.get_setup_type() != Setups.TRAIN_ONLY:
            metrics = self.evaluate(persist=persist)
            self.logger.notify("\n---------------------------")
            self.logger.notify("Evaluation Summary...")
            self.logger.notify("---------------------------\n")
            self.logger.error(metrics)
            self.logger.notify("\n---------------------------")
            self.logger.notify("---------------------------\n")

        saved_dirs = self.save(save_predictions=save_predictions)
        self.logger.notify("\n---------------------------")
        self.logger.notify("Results Saved To...")
        self.logger.notify("---------------------------\n")
        self.logger.error(saved_dirs)
        self.logger.notify("\n---------------------------")
        self.logger.notify("---------------------------\n")
        return saved_dirs

    @classmethod
    def flush(cls):
        gc.collect()
        torch.cuda.empty_cache()
        for proc in psutil.process_iter(attrs=['pid', 'status']):
            if proc.info['status'] == psutil.STATUS_ZOMBIE:
                try:
                    os.kill(proc.info['pid'], signal.SIGKILL)
                except (PermissionError, ProcessLookupError):
                    pass

    def _initialize_metric_evaluator(self):
        self.metric_evaluator = ClassificationMetricEvaluator(
            self.label_handler,
            main_metric_name=self.setup.get_main_metric_name(),
            complementary_metrics_names=self.setup.get_completion_metrics_names(),
            top_k=self.setup.get_top_k(),
        )

    def _initialize_model_config(self):
        context_init = self.label_handler.get_context_initialization() if self.setup.get_use_dataset_context_init() else None
        
        self.model_config = ModelConfigManager.get_config(
            backbone_name=self.setup.get_backbone_name(),
            source=self.setup.get_source(),
            trainer_name=self.setup.get_trainer_name(),
            labels=self.label_handler.get_labels(),
            context_initialization=context_init,
            label_id_prompt_id_mapping=self.label_handler.get_label_id_prompt_id_mapping(),
            do_augmentation=self.setup.get_do_augmentation(),
            fine_to_coarse_label_id_mapping=self.label_handler.get_fine_to_coarse_label_id_mapping(
                coarse_column_name=self.setup.get_coarse_column_name(),
            ),
            loss_type=self.setup.get_loss_type(),
            loss_kwargs=self.setup.get_loss_kwargs(),
        )
    
    def _initialize_trainer_config(self):
        optimization_configs = TrainersOptimConfig.get_config(
            self.setup.get_backbone_name(),
            self.setup.get_trainer_name(),
        )
        self._initialize_metric_evaluator()

        self.trainer_args = TrainingArguments(
            **TrainingArgumentsConfig.get_config(
                setup_type=self.setup.get_setup_type(),
                precision_dtype=self.setup.get_precision_dtype(),
                eval_full_precision=self.setup.get_eval_full_precision(),
                train_full_precision=self.setup.get_train_full_precision(),
                auto_find_batch_size=self.setup.get_auto_batch_size(),
                train_batch_size=self.setup.get_train_batch_size(),
                eval_batch_size=self.setup.get_eval_batch_size(),
                num_epochs=self.setup.get_num_epochs(),
                metric_for_best_model=self.metric_evaluator.get_main_metric_name(),
                label_names=MultiModalDataCollator.get_label_names(),
                greater_is_better=self.metric_evaluator.is_greater_better(),
                tf32=self.setup.get_enable_tensor_float(),
            ),
            **optimization_configs['optimizer'],
            **optimization_configs['lr_scheduler'],
        )

        callbacks = [] 
        if self.setup.get_setup_type() != Setups.EVAL_ONLY:
            callbacks = [EarlyStoppingCallback(**optimization_configs['early_stopping'])]
        
        self.trainer_callbacks = callbacks

    def _initialize_trainer(self, train_dataset=None, eval_dataset=None, data_collator=None):
        self._initialize_trainer_config()
        self.trainer = Trainer(
            model=self.model,
            args=self.trainer_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.metric_evaluator,
            callbacks=self.trainer_callbacks,
        )

    def _initialize_label_configs(self, **kwargs):
        self.annotations_config = AnnotationsConfig.get_config(dataset_name=self.setup.get_dataset_name())
        self.label_handler = (
            LabelHandleFactory.create_from_config(self.annotations_config)
            .set_prompt_mode(is_soft=self.setup.get_is_soft())
            .filter_labels(filter_dict=self.setup.get_annotations_key_value_criteria())
        )
        if self.setup.get_label_column_name() not in [None, 'default']:
            self.label_handler.update_label(label_source=self.setup.get_label_column_name())

        self.label_handler.config_prompts(**kwargs)
        num_labels = self.label_handler.get_num_labels()
        self.setup.set_top_k(min(num_labels, self.setup.get_top_k()))

    def _initialize_data_config(self, stages=None):
        stages = stages if stages is not None else self._get_stages()
        for stage in stages:
            for modality_idx in Modalities.get_values():
                self.stage_handlers_configs[modality_idx][stage] = self._create_handler_config(modality_idx, stage)
                modality = ModalityManager.get_singleton_modality(modality_idx, stage) 
        
    def _initialize_configs(self, **kwargs):
        self._initialize_label_configs(**kwargs)
        self._initialize_model_config()
        self._initialize_data_config()

        self.logger.notify("\n---------------------------")
        self.logger.notify("Labels Summary...")
        self.logger.notify("---------------------------\n")
        self.label_handler.show(logging_fn=self.logger.info)
        self.logger.notify("\n---------------------------")
        self.logger.notify("---------------------------\n")

    def setup_data(self, stages=None, persist=False):
        self.logger.notify("---------------------------")
        self.logger.notify("Loading Datasets...")
        self.logger.notify("---------------------------\n")
        stages = stages if stages is not None else self._get_stages()
        for stage in stages:
            for modality_idx in Modalities.get_values():
                self.stage_handlers[modality_idx][stage] = self._create_handler(modality_idx, stage)
                modality = ModalityManager.get_singleton_modality(modality_idx, stage)
                if persist:
                    self.stage_handlers[modality_idx][stage].persist()
                self.logger.info(
                    f'{modality_idx} @ {stage} ({modality.get_type()} -> {modality.get_status()}) Dataset Loaded.',
                )

        self.logger.notify("\n---------------------------")
        self.logger.notify("Datasets Summary...")
        self.logger.notify("---------------------------\n")
        for stage in stages:
            for modality_idx in Modalities.get_values():
                modality = ModalityManager.get_singleton_modality(modality_idx, stage)
                features = self.stage_handlers[modality_idx][stage].get_dataset_features()
                self.logger.info(
                    f'{modality_idx} @ {stage} ({modality.get_type()} -> {modality.get_status()}):\n {features}\n',
                )

        self.logger.notify("\n---------------------------")
        self.logger.notify("---------------------------\n")

    def setup_model(self):
        labels_sample_count = None 
        if self.setup.get_setup_type() != Setups.EVAL_ONLY:
            labels_sample_count = torch.tensor(
                self.stage_handlers[Modalities.M1][Stages.TRAIN].get_class_count(
                    self.label_handler.get_label_id_column()
                )
            ).to(self.device)

        self.model_config.set_labels_sample_count(labels_sample_count)
        model = VLMFactory.from_pretrained(model_config=self.model_config).to(self.device)
        
        if self.setup.get_load_from_checkpoint():
            checkpoint_path = self.setup.get_model_checkpoint_path()
            self.logger.info("\n---------------------------")
            model.load(checkpoint_path)

            model = model.to(self.device)
            self.logger.info("---------------------------\n")
            if self.setup.get_is_soft():
                model.register_new_labels(
                    self.label_handler.get_labels(),
                    self.label_handler.get_label_id_prompt_id_mapping().to(self.device),
                    use_learned_contex=True,
                )

        self.model = model.to(self.device)
        self.logger.notify("\n---------------------------")
        self.logger.notify("Model Summary...")
        self.logger.notify("---------------------------\n")
        self.model.show(logging_fn=self.logger.info)
        self.logger.notify("\n---------------------------")
        self.logger.notify("---------------------------\n")

    def _prepare_data(self, stage, persist=False):
        for modality_idx in Modalities.get_values():
            if persist:
                self.stage_handlers[modality_idx][stage].persist()
            self.stage_handlers[modality_idx][stage] = self._process_handler(key=modality_idx, stage=stage)

        sampling_arguments = {} if stage != Stages.TRAIN else dict(
            sampling_type=self.setup.get_sampling_type(),
            sampling_strategy=self.setup.get_sampling_strategy(),
            sampling_column=self.label_handler.get_label_id_column(),
            sampling_kwargs=self.setup.get_sampling_kwargs(),
        )
        handlers = self._get_stage_handlers(stage=stage)
        datasets = handlers[Modalities.M1].get_dataset(
            split_size=self.setup.get_validation_size() if stage == Stages.TRAIN else None,
            random_state=self.setup.get_random_state() if stage == Stages.TRAIN else None,
            keep_necessary_cols_only=True,  **sampling_arguments,
        )
        return datasets

    def _prepare_collator(self, stage, collate_all_m2_samples=False):
        m1_processor, m2_processor = [
            self._create_handler_augmentor(modality_idx, stage) for modality_idx in Modalities.get_values()
        ]

        handlers = self._get_stage_handlers(stage=stage)
        data_collator = DataCollatorFactory.create_multimodal_collator(
            self.label_handler.get_class_id_label_id_adj_matrix(),
            m1_dataset_handler=handlers[Modalities.M1],
            m2_dataset_handler=handlers[Modalities.M2],
            m1_processor=m1_processor,
            m2_processor=m2_processor,
            collate_all_m2_samples=collate_all_m2_samples,
        )
        return data_collator

    def _sync_setup(self, stage):
        current_model_type = self.setup.get_model_type()
        current_setup_type = self.setup.get_setup_type()
        is_few_shot = self.setup.get_n_shots() is not None

        if stage == Stages.TRAIN:
            eligible_model_types = [ModelType.FULL_TRAINED, ModelType.FEW_SHOT]
            eligible_setup_types = [Setups.TRAIN_ONLY, Setups.FULL]

        elif stage == Stages.EVAL:
            eligible_model_types = ModelType.get_values()
            eligible_setup_types = [Setups.EVAL_ONLY, Setups.FULL]

        alternative_model_type = eligible_model_types[1] if is_few_shot else eligible_model_types[0]
        alternative_setup_type = eligible_setup_types[1]
        
        if current_model_type not in eligible_model_types:
            new_model_type = alternative_model_type
            self.logger.notify(f"Model type changed from '{current_model_type}' to '{alternative_model_type}'")
            self.setup.set_model_type(new_model_type)

        if current_setup_type not in eligible_setup_types:
            new_setup_type = alternative_setup_type
            self.logger.notify(f"Setup type changed from '{current_setup_type}' to '{alternative_setup_type}'")
            self.setup.set_setup_type(new_setup_type)        

    def _train_model(self, train_dataset, eval_dataset, data_collator):
        self._initialize_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        with torch.autocast(device_type=self.device_type):
            self.trainer.train()

        self.model = self.trainer.model

    def _evaluate_model(self, eval_dataset, data_collator):
        self._initialize_trainer(eval_dataset=eval_dataset, data_collator=data_collator)
        with torch.autocast(device_type=self.device_type):
            metrics = self.trainer.evaluate()
        return metrics

    def _run_stage(self, stage, collate_all_m2_samples=False, persist=False):
        datasets = self._prepare_data(stage=stage, persist=persist)
        data_collator = self._prepare_collator(stage=stage, collate_all_m2_samples=collate_all_m2_samples)
        self._sync_setup(stage=stage)
        if stage == Stages.TRAIN:
            self._train_model(
                train_dataset=datasets[Stages.TRAIN],
                eval_dataset=datasets[Stages.EVAL],
                data_collator=data_collator,
            )
            metrics = None
        elif stage == Stages.EVAL:
            metrics = self._evaluate_model(eval_dataset=datasets, data_collator=data_collator)

        self.flush()
        return metrics

    def train(self, collate_all_m2_samples=False, persist=False):
        return self._run_stage(stage=Stages.TRAIN, collate_all_m2_samples=collate_all_m2_samples, persist=persist)

    def evaluate(self, persist=False):
        metrics = self._run_stage(stage=Stages.EVAL, collate_all_m2_samples=True, persist=persist)
        return metrics

    def save_model(self, directory=None):
        try:
            saved_dir = self.model.save(self.setup, directory=directory)
            return saved_dir
        except Exception as e:
            self.logger.info(f"Failed to save the model: {e}")

    def save_metrics(self, directory=None, save_predictions=False):
        try:
            saved_dir = self.metric_evaluator.save(self.setup, directory=directory, save_predictions=save_predictions)
            return saved_dir
        except Exception as e:
            self.logger.info(f"Failed to save metrics: {e}")
    
    def save(self, model_directory=None, metric_directory=None, save_predictions=False):
        saved_dirs_dict = {}
        if self.setup.get_setup_type() != Setups.EVAL_ONLY:
            saved_dirs_dict['model_dir'] = self.save_model(directory=model_directory)

        if self.setup.get_setup_type() != Setups.TRAIN_ONLY:
            saved_dirs_dict['metrics_dir'] = self.save_metrics(directory=metric_directory, save_predictions=save_predictions)

        saved_dirs_dict['setup_dir']= self.setup.save()
        return saved_dirs_dict

    def _create_handler_config(self, key, stage):
        modality = ModalityManager.get_singleton_modality(key=key, stage=stage)
        if modality.get_type() == ModalityType.IMAGE:
            perform_augmentation = modality.get_perform_augmentation()
            load_raw_data = perform_augmentation or modality.get_requires_grad()
            try:
                image_config = ImageDatasetConfig.get_config(
                    self.setup,
                    split=self.setup.get_split(stage),
                    data_type=DataStatus.RAW if load_raw_data else DataStatus.EMBEDDING,
                )
            except Exception as e:
                self.logger.info(f'Failed loading embedding! Loading raw data...: {e}')
                image_config = ImageDatasetConfig.get_config(self.setup, split=stage, data_type=DataStatus.RAW)
            return image_config
        elif modality.get_type() == ModalityType.TEXT:
            text_config = AnnotationsConfig.get_config(dataset_name=self.setup.get_dataset_name())
            return text_config
        else:
            raise ValueError('Wrong modality type')

    def _create_handler(self, key, stage):
        if not (key in self.stage_handlers_configs and stage in self.stage_handlers_configs[key]):
            raise AssertionError('Wrong initialization.')
        
        handler_config = self.stage_handlers_configs[key][stage]
        modality = ModalityManager.get_singleton_modality(key=key, stage=stage, status=handler_config['data_type'])
        
        if modality.get_type() == ModalityType.IMAGE:
            image_handler = ImageHandlerFactory.create_from_config(
                key,
                stage,
                handler_config,
                to_keep_ids=self.label_handler.get_class_ids(),
                n_shots=self.setup.get_n_shots() if stage == Stages.TRAIN else None,
            )

            image_handler.map_dataset_identifier(
                mapping=self.label_handler.get_class_id_label_id_mapping(),
                store_in_col_name=self.label_handler.get_label_id_column(),
            )
            image_handler.post_init()
            return image_handler

        elif modality.get_type() == ModalityType.TEXT:
            prompts_df = self.label_handler.get_prompts_df()
            text_handler = TextHandlerFactory.create_from_df(
                key,
                stage,
                prompts_df,
                handler_config,
            )
            return text_handler
        else:
            raise ValueError('Wrong modality type')

    def _create_handler_augmentor(self, key, stage):
        self.stage_handlers[key][stage]
        modality = ModalityManager.get_singleton_modality(key=key, stage=stage)
        if not modality.get_perform_augmentation():
            return None

        if modality.get_type() == ModalityType.IMAGE and modality.is_raw():
            modality.update_status(modality.get_next_status())
            return self.model.get_image_processor()
        return None


    def _process_handler(self, key=None, stage=None, handler=None):
        key = key or handler.get_modality().get_key()
        stage = stage or handler.get_modality().get_stage()
        handler = self.stage_handlers[key][stage] if handler is None else handler
        if not self._should_process_modality(key, stage):
            return handler

        self.model.eval()
        modality = ModalityManager.get_singleton_modality(key=key, stage=stage)

        with torch.no_grad(), torch.autocast(device_type=self.device_type, dtype=self.precision_dtype):
            self.logger.info(f'Embedding {key} @ {stage}...')
            handler = handler.to_embedding(
                self.model.get_embedding_fn_for_modality(modality),
                batch_size=self.setup.get_preprocess_batch_size(),
                keep_necessary_cols_only=False,
            )
            self.flush()
        return handler

    def _should_process_modality(self, key, stage):
        modality = ModalityManager.get_singleton_modality(key=key, stage=stage)
        skip_preprocess = modality.get_perform_augmentation() or modality.get_requires_grad() or modality.is_embedded()
        if skip_preprocess:
            return False
        if modality.get_requires_preprocess():
            return True
        return False

    def _get_stage_handlers(self, stage):
        return {modality_idx: self.stage_handlers[modality_idx][stage] for modality_idx in Modalities.get_values()}

    def _get_stages(self):
        if self.setup.get_setup_type() == Setups.EVAL_ONLY:
            return [Stages.EVAL]
        if self.setup.get_setup_type() == Setups.TRAIN_ONLY:
            return [Stages.TRAIN]
        return [Stages.TRAIN, Stages.EVAL]

    def _get_stage_modalities(self, stage):
        return {modality_idx: ModalityManager.get_singleton_modality(key=modality_idx, stage=stage) for modality_idx in Modalities.get_values()}

    def get_model(self):
        return self.trainer.model if self.trainer is not None else self.model

    def get_metrics(self, main_metric_only=True):
        metrics = self.metric_evaluator.get_metrics(main_metric_only=main_metric_only)
        return metrics

    def tear_down(self):
        self.logger.info("Tearing down resources and cleaning up...")

        if hasattr(self, 'tensorboard_connector') and self.tensorboard_connector:
            self.tensorboard_connector.stop()
            delattr(self, 'tensorboard_connector')
            self.logger.info("TensorBoard has been stopped and attribute removed.")

        attributes_to_delete = [
            'model', 'trainer', 'trainer_args', 'trainer_callbacks', 'metric_evaluator',
            'annotations_config', 'label_handler', 'model_config', 'stage_handlers',
            'stage_handlers_configs',
        ]

        for attr in attributes_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)
                self.logger.info(f"Deleted attribute {attr}")

        ModalityManager.flush()
        self.flush()
        self.logger.info("Resources have been freed up!")

    def show(self):
        self.logger.info(str(self))

    def __str__(self):
        attrs_names = [
            'setup',
            'device_type',
            'model',
            'metric_evaluator',
            'stage_handlers',
            'tensorboard_connector',
        ]
        attributes = [
            f"{attr}={getattr(self, attr)!s}" for attr in attrs_names if getattr(self, attr, None) is not None
        ]
        attributes_str = '\n'.join(attributes)
        indented_attributes_str = textwrap3.indent(attributes_str, '    ')  
        return f"{self.__class__.__name__}(\n{indented_attributes_str}\n)"
