from config.enums import Sources, Trainers
from model.baseline_models.coop import COOP
from model.baseline_models.huggingface_clip import HuggingFaceCLIP
from model.baseline_models.openai_clip import OpenAICLIP


class VLMFactory:
    @staticmethod
    def from_pretrained(
        model_config,
        **kwargs,
    ):
        source = model_config['source']
        vlm_class = None
        if source == Sources.HUGGINGFACE:
            vlm_class = HuggingFaceCLIP
        elif source == Sources.OPEN_AI:
            if model_config['trainer_name'] == Trainers.CLIP:
                vlm_class = OpenAICLIP
            elif model_config['trainer_name'] in [Trainers.COOP]:
                vlm_class = COOP
            else:
                raise ValueError('Model has not been implemented yet!')

        vlm = vlm_class.from_pretrained(model_config=model_config, **kwargs)
        return vlm
