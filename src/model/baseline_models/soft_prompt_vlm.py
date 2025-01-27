import torch

from model.clip_modeling.openai import clip
from model.clip_modeling.openai.clip import get_image_transforms
from model.vlm_wrappers import OpenAIWrapper


class SoftPromptVLM(OpenAIWrapper):
    def __init__(self, *args, **kwargs):
        self.soft_prompting_config = kwargs.pop('soft_prompting_config', {})
        super().__init__(*args, **kwargs)

    skip_keys_in_load = [
        'prompt_learner.tokenized_prompts',
    ]
    
    def register_new_labels(self, labels, label_id_prompt_id_mapping, use_learned_contex=True):
        raise NotImplementedError
    
    @staticmethod
    def fetch_model(model_url, device='cpu'):
        url = clip._MODELS[model_url]
        model_os_path = clip._download(url)
        model = None
        try:
            model = torch.jit.load(model_os_path, map_location=device).eval()
            state_dict = None
    
        except RuntimeError:
            state_dict = torch.load(model_os_path, map_location=device)
        
        return model, state_dict
        
    @staticmethod
    def build_model(model_url, trainer, soft_prompting_config, device='cpu'):
        model, state_dict = SoftPromptVLM.fetch_model(model_url)
        design_details = {
            'trainer': trainer,
            'vision_depth': soft_prompting_config.get_image_depth(),
            'language_depth': soft_prompting_config.get_text_depth(),
            'vision_ctx': soft_prompting_config.get_image_n_context(),
            'language_ctx': soft_prompting_config.get_text_n_context(),
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model, get_image_transforms(model)
