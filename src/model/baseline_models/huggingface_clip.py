import torch
from transformers import (
    CLIPConfig,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from model.vlm_wrappers import HuggingFaceWrapper


class HuggingFaceCLIP(HuggingFaceWrapper):
    @staticmethod
    def from_pretrained(model_config, **kwargs):
        model_path = model_config['model_url']
        config = CLIPConfig.from_pretrained(model_path)
        image_tower = CLIPVisionModelWithProjection.from_pretrained(model_path)
        text_tower = CLIPTextModelWithProjection.from_pretrained(model_path)
        torch.nn.Parameter(torch.tensor(config.logit_scale_init_value))
        return HuggingFaceCLIP(text_tower, image_tower, config, model_config=model_config, **kwargs)
