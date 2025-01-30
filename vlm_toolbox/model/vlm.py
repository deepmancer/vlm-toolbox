import copy
import os

import torch
from torch import nn

from config.enums import DataStatus, Modalities, ModalityType, Stages
from config.modality import ModalityManager
from loss.factory import LossFactory
from model.data_classes import VLMOutput
from model.processors.factory import ImageProcessorFactory, TextProcessorFactory
from util.torch_helper import Aggregator, describe_model, get_model_device


class VLM(nn.Module):
    skip_keys_in_load = []
    _keys_to_ignore_on_save = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.output_dataclass = VLMOutput
        self.model_config = kwargs.get('model_config')

        self.image_modality = ModalityManager.get_modality(ModalityType.IMAGE, source=self.model_config['source'])
        self.text_modality = ModalityManager.get_modality(ModalityType.TEXT, source=self.model_config['source'])
        
        self.aggregator = Aggregator(self.model_config['feature_aggregation_config'])

        self.image_processor = ImageProcessorFactory.create(self.model_config)
        self.text_processor = TextProcessorFactory.create(self.model_config)

        self.eval()

    @property
    def m_config(self):
        return {
            'm1': self.m1_config,
            'm2': self.m2_config,
        }

    @property
    def m1_config(self):
        return ModalityManager.get_singleton_modality(key=Modalities.M1, stage=self.stage)

    @property
    def m2_config(self):
        return ModalityManager.get_singleton_modality(key=Modalities.M2, stage=self.stage)

    @property
    def device(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    def train(self, mode=True):
        super().train(mode)
        self.stage = Stages.TRAIN if mode else Stages.EVAL
        return self

    def eval(self):
        super().eval()
        self.stage = Stages.EVAL
        return self
    
    def show(self, logging_fn=print):
        logging_fn(str(self))
        describe_model(self, logging_fn=logging_fn)

    def state_dict(self, *args, **kwargs):
        raise NotImplementedError
    
    def load_state_dict(self, state_dict, strict=True, assign=False):
        raise NotImplementedError

    def load(self, checkpoint_path, strict=False, assign=False, logging_fn=print, **kwargs):
        try:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=strict, assign=assign, logging_fn=logging_fn, **kwargs)
            logging_fn(f"Model weights loaded successfully from {checkpoint_path}")
        except Exception as e:
            logging_fn(f"Failed to load model: {e}")
    
    def save(self, setup=None, directory=None, file_name="pytorch_model.bin", logging_fn=print):
        save_dir = directory or self.model_config.get_default_save_path(setup)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        base_name, ext = os.path.splitext(file_name)
        model_path = os.path.join(save_dir, file_name)
        counter = 1
        while os.path.exists(model_path):
            model_path = os.path.join(save_dir, f"{base_name}_{counter}{ext}")
            counter += 1

        try:
            torch.save(self.state_dict(), model_path)
            logging_fn(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logging_fn(f"Failed to save model: {e}")
        return None

    def get_processor(self, modality_type):
        if modality_type == ModalityType.IMAGE:
            return self.image_processor
        elif modality_type == ModalityType.TEXT:
            return self.text_processor
        raise ValueError(f"Invalid modality type: {modality_type}")
    
    def get_text_preprocessor(self):
        return self.text_processor

    def get_logit_scale(self):
        raise NotImplementedError

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        raise NotImplementedError
    
    def preprocess_text(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            inputs = [v for k, v in inputs.items() if k in self.text_modality.get_raw_keys()][0]
            
        return self.text_processor(inputs)

    def get_image_processor(self):
        return self.image_processor

    def preprocess_image(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            inputs = [v for k, v in inputs.items() if k in self.image_modality.get_raw_keys()][0]
        do_augmentation = (self.stage == Stages.TRAIN) and self.model_config.get_do_augmentation()
        return self.image_processor(inputs, do_augmentation=do_augmentation)

    def compute_processed_text_features(self, *args, **kwargs):
        raise NotImplementedError

    def compute_processed_image_features(self, *args, **kwargs):        
        raise NotImplementedError

    def compute_raw_text_features(self, inputs, **kwargs):
        preprocessed_texts = {k: v.to(get_model_device(self)) for k, v in self.preprocess_text(inputs).items()}
        output = self.compute_processed_text_features(preprocessed_texts)
        return output

    def compute_raw_image_features(self, inputs, **kwargs):
        preprocessed_images = {k: v.to(get_model_device(self)) for k, v in self.preprocess_image(inputs).items()}
        output = self.compute_processed_image_features(preprocessed_images)
        return output
    
    def prepare_output(self, loss=None, return_loss=False, return_dict=False, **required_kwargs):
        output = self.output_dataclass(
            **required_kwargs,
        )
        if return_loss:
            output['loss'] =  loss
        if return_dict:
            return output
        return output.to_tuple()

    def get_loss_fn(self, **kwargs):
        loss_config = copy.deepcopy(self.model_config.get_loss_config().get_loss_config(self.stage))
        loss_class_name = loss_config.get('loss_class')
        loss_wrapper_name = loss_config.get('wrapper_class', None)
        loss_kwargs = loss_config['kwargs']
        labels_sample_count = self.model_config.get_labels_sample_count()
        fine_to_coarse_mapping = None
        if loss_wrapper_name is not None:
            fine_to_coarse_mapping = self.model_config.get_fine_to_coarse_label_id_mapping().to(self.device)

        is_multi_class = kwargs.pop('is_multi_class', False)

        loss_kwargs = {**kwargs, **loss_kwargs}
        return LossFactory.create_loss(
            loss_class_name,
            wrapper_class=loss_wrapper_name,
            fine_to_coarse_mapping=fine_to_coarse_mapping,
            labels_sample_count=labels_sample_count,
            is_multi_class=is_multi_class,
            **loss_kwargs,
        )

    def compute_loss(self, is_multi_class=True, **kwargs):   
        loss_fn = self.get_loss_fn(is_multi_class=is_multi_class)
        return loss_fn(**kwargs)    
    
    def forward_modality(self, modality_dict, modality, **kwargs):
        m_batch_values, inverse_index = modality_dict['features'], modality_dict.get('inverse_index', None)
        embedding_key = modality.get_embedding_keys()[0]
        if modality.is_embedded():
            m_features = m_batch_values[embedding_key]
        else:
            embedding_input_keys = modality.get_next_values()
            m_embedding_fn = self.get_embedding_fn_for_modality(modality)
            
            if not modality.get_requires_grad():
                with torch.no_grad():
                    m_features = m_embedding_fn(
                        {k: m_batch_values[k] for k in embedding_input_keys}
                    )[embedding_key]
            else:
                m_features = m_embedding_fn(
                    {k: m_batch_values[k] for k in embedding_input_keys}
                )[embedding_key]

        if inverse_index is not None:
            m_features = self.aggregator.aggregate('embedding', m_features, inverse_index)

        normalized_m_features = m_features / m_features.norm(p=2, dim=-1, keepdim=True)
        return m_features, normalized_m_features

    def get_embedding_fn_for_modality(self, modality):
        status = modality.get_status()
        if modality.get_type() == ModalityType.TEXT:
            if status == DataStatus.PREPROCESSED:
                return self.compute_processed_text_features
            elif status == DataStatus.RAW and modality.get_requires_preprocess():
                return self.compute_raw_text_features
            elif status == DataStatus.RAW:
                return self.compute_processed_text_features
            elif status == DataStatus.EMBEDDING:
                return lambda x, *args, **kwargs: x
            else:
                raise ValueError

        if modality.get_type() == ModalityType.IMAGE:
            if status == DataStatus.PREPROCESSED:
                return self.compute_processed_image_features
            elif status == DataStatus.RAW and modality.get_requires_preprocess():
                return self.compute_raw_image_features
            elif status == DataStatus.RAW:
                return self.compute_processed_image_features
            elif status == DataStatus.EMBEDDING:
                return lambda x, *args, **kwargs: x
            else:
                raise ValueError
            
    def forward(
        self,
        return_loss=True,
        return_dict=True,
        **kwargs,
    ):
        m1, m2, is_multi_class = kwargs.get(Modalities.M1), kwargs.get(Modalities.M2), kwargs.get('is_multi_class', False)
        m1_config, m2_config = self.m1_config, self.m2_config

        m1_features, normalized_m1_features = self.forward_modality(m1, m1_config)
        m2_features, normalized_m2_features = self.forward_modality(m2, m2_config)

        with torch.no_grad():
            m1_m2_logits = self.get_logit_scale() * torch.matmul(normalized_m1_features, normalized_m2_features.t())

        if return_loss:
            labels = kwargs.get('labels', None)
            loss = self.compute_loss(
                is_multi_class=is_multi_class,
                normalized_m1_features=normalized_m1_features,
                normalized_m2_features=normalized_m2_features,
                m1_ids=m1['ids'],
                m2_ids=m2['ids'],
                labels=labels,
                temperature=self.get_logit_scale(),
            )
        else:
            loss = None

        
        return self.prepare_output(
            loss,
            return_loss,
            return_dict,
            m1_ids= m1['ids'],
            m2_ids=m2['unique_ids'] if m2['unique_ids'] is not None else m2['ids'],
            m1_embeds=m1_features,
            m2_embeds=m2_features,
            m1_m2_logits=m1_m2_logits,
        )

    def __str__(self):
        representation_parts = [
            f'{self.model_config.get_trainer_name()} - {self.model_config.get_backbone_name()}',
            f'Logit scale: {self.get_logit_scale():.4f}',
            f'Modalities @ {self.stage.upper()}',
            '',
            f'M1: {self.m1_config}',
            f'M2: {self.m2_config}',
            '',
        ]
        return '\n'.join(representation_parts)  

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items() if key in self.__dict__)
        return f"{class_name}({attributes})"
