from config.base import BaseConfig
from config.enums import ModalityType, Trainers


DEFAULT_SOFT_PROMPTING_CONFIG = {
    ModalityType.IMAGE: {
        'depth': 0,
        'n_context': 0,
        'context_initialization': None,
    },
    ModalityType.TEXT: {
        'depth': 0,
        'n_context': 0,
        'context_initialization': 'it is a photo of',
        'prompts': {
            'class_token_position': 'end',
        },
    },
}

class SoftPrompt:
    """A class to manage soft prompting configuration for different modalities."""
    def __init__(self, config):
        self.config = {**DEFAULT_SOFT_PROMPTING_CONFIG, **config}

    def get_image_depth(self):
        return self.config[ModalityType.IMAGE]['depth']

    def set_image_depth(self, depth):
        self.config[ModalityType.IMAGE]['depth'] = depth

    def get_image_n_context(self):
        return self.config[ModalityType.IMAGE]['n_context']

    def set_image_n_context(self, n_context):
        self.config[ModalityType.IMAGE]['n_context'] = n_context

    def get_image_context_initialization(self):
        return self.config[ModalityType.IMAGE]['context_initialization']

    def set_image_context_initialization(self, initialization):
        self.config[ModalityType.IMAGE]['context_initialization'] = initialization

    def get_text_depth(self):
        return self.config[ModalityType.TEXT]['depth']

    def set_text_depth(self, depth):
        self.config[ModalityType.TEXT]['depth'] = depth

    def get_text_n_context(self):
        return self.config[ModalityType.TEXT]['n_context']

    def set_text_n_context(self, n_context):
        self.config[ModalityType.TEXT]['n_context'] = n_context

    def get_text_context_initialization(self):
        return self.config[ModalityType.TEXT]['context_initialization']

    def set_text_context_initialization(self, initialization):
        self.config[ModalityType.TEXT]['context_initialization'] = initialization

    def get_prompt_prefix(self):
        if self.get_text_context_initialization():
            return '"a photo of a'

        return None
        
    def get_class_token_position(self):
        return self.config[ModalityType.TEXT]['prompts']['class_token_position']

    def set_class_token_position(self, position):
        self.config[ModalityType.TEXT]['prompts']['class_token_position'] = position

    def __repr__(self):
        return f"SoftPrompt(config={self.config})"

class SoftPromptingConfig(BaseConfig):
    config = {
        Trainers.COOP: {
            ModalityType.TEXT: {
                'depth': 9,
                'n_context': 16,
                'context_initialization': None,
                'prompts': {
                    'class_token_position': 'end',
                },
            },
        },
    }

    @staticmethod
    def get_zero_shot_config():
        return SoftPrompt(DEFAULT_SOFT_PROMPTING_CONFIG)

    @staticmethod
    def get_config(trainer_name=Trainers.COOP):
        assert SoftPromptingConfig.is_valid(trainer_name), 'Design not available! Choose from the available trainers.'
        return SoftPrompt(SoftPromptingConfig.get(trainer_name))
