from config.enums import Sources
from model.processors.image_processors import (
    HuggingFaceImageProcessor,
    OpenAIImageProcessor,
)
from model.processors.text_processors import (
    HuggingFaceTextProcessor,
    OpenAITextProcessor,
)


class TextProcessorFactory:
    @staticmethod
    def create(model_config):
        source = model_config['source']
        if source == Sources.HUGGINGFACE:
            return HuggingFaceTextProcessor(model_config)
        elif source == Sources.OPEN_AI:
            return OpenAITextProcessor(model_config)

        raise ValueError(f'Invalid Source')

class ImageProcessorFactory:
    @staticmethod
    def create(model_config):
        source = model_config['source']
        if source == Sources.HUGGINGFACE:
            return HuggingFaceImageProcessor(model_config)
        elif source == Sources.OPEN_AI:
            return OpenAIImageProcessor(model_config)

        raise ValueError(f'Invalid Source')
