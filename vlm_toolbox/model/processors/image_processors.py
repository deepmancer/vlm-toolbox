import torch
from timm.data import create_transform
from transformers import CLIPProcessor


class HuggingFaceImageProcessor:
    def __init__(self, model_config):
        self.processor = CLIPProcessor.from_pretrained(model_config['model_url'])

    def __call__(self, inputs, **kwargs):
        preprocessed_images = self.processor(images=inputs, return_tensors='pt')
        return {
            'pixel_values': preprocessed_images['pixel_values'],
        }

class OpenAIImageProcessor:
    def __init__(self, model_config):
        self.processor = create_transform(**model_config['image_transforms_config'])
        if model_config.get_do_augmentation():
            self.augmentor = create_transform(**model_config['image_augmentation_config'])

    def __call__(self, inputs, do_augmentation=False, **kwargs):
        processor = self.augmentor if do_augmentation else self.processor
        preprocessed_images = torch.stack([processor(input_.convert('RGB')) for input_ in inputs])
        return {
            'pixel_values': preprocessed_images,
        }
