import torch
from torch import nn

from model.vlm import VLM


class OpenAIWrapper(VLM):    
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=False, assign=False, logging_fn=print):
        skip_keys_in_load = self.skip_keys_in_load
        state_dict_keys = list(state_dict.keys())
        for skip_key in skip_keys_in_load:
            for state_key in state_dict_keys:
                if skip_key in state_key:
                    strict = False
                    del state_dict[state_key]

        casted_state_dict = {}
        for name, param in self.model.named_parameters():
            if name in state_dict:
                casted_state_dict[name] = state_dict[name].to(device=param.device, dtype=param.dtype)
            else:
                casted_state_dict[name] = param.to(param.device)
    
        return self.model.load_state_dict(casted_state_dict, strict=strict, assign=assign)

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    @property
    def dtype(self):
        for name, param in self.model.named_parameters():
            if 'ln' not in name and 'logit_scale' not in name and param.requires_grad:
                return param.dtype

    def get_logit_scale(self):
        return self.model.logit_scale.exp()

    def compute_processed_text_features(self, inputs={}, **kwargs):
        encoder_inputs = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs
        text_features = self.model.encode_text(encoder_inputs)
        return {
            self.text_modality.get_embedding_keys()[0]: text_features,
        }

    def compute_processed_image_features(self, inputs={}, **kwargs):
        encoder_inputs = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs
        image_features = self.model.encode_image(encoder_inputs)
        return {
            self.image_modality.get_embedding_keys()[0]: image_features,
        }
    

class HuggingFaceWrapper(VLM):
    def __init__(self, text_tower, image_tower, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = self.config.text_config.hidden_size
        self.vision_embed_dim = self.config.vision_config.hidden_size
        self.hidden_size = self.text_embed_dim
        self.config.hidden_size = self.hidden_size
        
        self.image_tower = image_tower
        self.text_tower = text_tower

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    def state_dict(self, *args, **kwargs):
        return {
            'image_tower': self.image_tower.state_dict(),
            'text_tower': self.text_tower.state_dict(),
            'logit_scale': self.logit_scale,
        }

    def load_state_dict(self, state_dict, strict=True, assign=False, extra_skip_keys=[]):
        skip_keys_in_load = self.skip_keys_in_load + extra_skip_keys
        if len(skip_keys_in_load):
            for key in ['image_tower', 'text_tower', 'logit_scale']:
                state_dict[key] = {
                    k: v for k, v in state_dict[key].items() if k not in skip_keys_in_load
                }
            strict = False

        it_response = self.image_tower.load_state_dict(state_dict['image_tower'], strict=strict, assign=assign)
        tt_response = self.text_tower.load_state_dict(state_dict['text_tower'], strict=strict, assign=assign)
        self.logit_scale = state_dict['logit_scale']
        return {'image_tower': it_response, 'text_tower': tt_response}

    @property
    def device(self):
        return next(iter(self.image_tower.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.image_tower.parameters())).dtype
    
    def get_logit_scale(self):
        return self.logit_scale.exp()

    def compute_processed_text_features(self, inputs={}, **kwargs):
        model_inputs = {k: inputs[k] for k in self.text_modality.get_preprocessed_keys()}
        key = self.text_modality.get_embedding_keys()[0]
        return {
            key: self.text_tower(**model_inputs, **kwargs)[key]
        }

    def compute_processed_image_features(self, inputs={}, **kwargs):
        model_inputs = {k: inputs[k] for k in self.image_modality.get_preprocessed_keys()}
        key = self.image_modality.get_embedding_keys()[0]
        return {
            key: self.image_tower(**model_inputs, **kwargs)[key]
        }
