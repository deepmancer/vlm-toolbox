import clip
from transformers import CLIPTokenizerFast


class HuggingFaceTextProcessor:
    def __init__(self, model_config):
        self.processor = CLIPTokenizerFast.from_pretrained(model_config['model_url'])

    def __call__(self, inputs, **kwargs):
        preprocessed_texts = self.processor(text=inputs, return_tensors='pt', padding='max_length', truncation=True, **kwargs)
        return {
            'input_ids':preprocessed_texts['input_ids'],
            'attention_mask':preprocessed_texts['attention_mask'],
        }

class OpenAITextProcessor:
    def __init__(self, model_config):
        self.processor = clip.tokenize

    def __call__(self, inputs, **kwargs):
        preprocessed_texts = self.processor(inputs, **kwargs)
        return {
            'input_ids': preprocessed_texts,
        }
