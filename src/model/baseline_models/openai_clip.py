import clip

from model.vlm_wrappers import OpenAIWrapper


class OpenAICLIP(OpenAIWrapper):
    @staticmethod
    def from_pretrained(model_config, **kwargs):
        model_path = model_config['model_url']
        model, _ = clip.load(model_path)
        for name, param in model.named_parameters():
            param.requires_grad_(False)

        return OpenAICLIP(model, model_config=model_config, **kwargs)
