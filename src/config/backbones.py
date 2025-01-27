from config.base import BaseConfig
from config.enums import (
    Backbones,
    CLIPBackbones,
    ImageBackbones,
    Sources,
    TextBackbones,
)


class BackboneConfig(BaseConfig):
    config = {
        Backbones.IMAGE: ImageBackbones,
        Backbones.TEXT: TextBackbones,
        Backbones.MULTIMODAL: CLIPBackbones,
    }

    @staticmethod
    def get_config(backbone_type=Backbones.MULTIMODAL):
        return BackboneConfig.get(backbone_type)

class BackboneURLConfig(BaseConfig):
    config = {
        Backbones.IMAGE: {
            ImageBackbones.DYNO_V2_GIANT: {
                Sources.HUGGINGFACE: 'facebook/dinov2-giant',
            },
        },
        Backbones.TEXT: {
            TextBackbones.ALL_MINILM_L6_V2: {
                Sources.HUGGINGFACE: 'sentence-transformers/all-MiniLM-L6-v2',
            },
            TextBackbones.ALL_MPNET_BASE_V2: {
                Sources.HUGGINGFACE: 'sentence-transformers/all-mpnet-base-v2',
            },
        },
        Backbones.MULTIMODAL: {
            CLIPBackbones.CLIP_VIT_B_32: {
                Sources.HUGGINGFACE: 'openai/clip-vit-base-patch32',
                Sources.OPEN_AI:'ViT-B/32',
            },
            CLIPBackbones.CLIP_VIT_B_16: {
                Sources.HUGGINGFACE: 'openai/clip-vit-base-patch16',
                Sources.OPEN_AI: 'ViT-B/16',
            },
            CLIPBackbones.CLIP_VIT_L_14: {
                Sources.HUGGINGFACE: 'openai/clip-vit-large-patch14',
                Sources.OPEN_AI: 'ViT-L/14',
            },
            CLIPBackbones.CLIP_VIT_L_14_336PX: {
                Sources.OPEN_AI: 'ViT-L/14-336px',
                Sources.HUGGINGFACE: 'openai/clip-vit-large-patch14-336',    
            },

            CLIPBackbones.CLIP_RESNET_50: {Sources.OPEN_AI: 'RN50'},
            CLIPBackbones.CLIP_RESNET_101: {Sources.OPEN_AI: 'RN101'},
            CLIPBackbones.CLIP_RESNET_50_4: {Sources.OPEN_AI: 'RN50x4'},
            CLIPBackbones.CLIP_RESNET_50_16: {Sources.OPEN_AI: 'RN50x16'},
            CLIPBackbones.CLIP_RESNET_50_64: {Sources.OPEN_AI: 'RN50x64'},
        }
    }

    @staticmethod
    def get_config(backbone_type=Backbones.MULTIMODAL, backbone_name=CLIPBackbones.CLIP_VIT_B_32, source=Sources.HUGGINGFACE):
        assert BackboneURLConfig.is_valid(backbone_type, strict=True), f'Backbone Type not available! Choose between: {list(BackboneURLConfig.config.keys())}'
        backbone_config = BackboneURLConfig.get(backbone_type, None)
        assert backbone_config[backbone_name] != None, f'Model not available! Choose between: {list(backbone_config.keys())}'
        model_config = backbone_config[backbone_name]
        assert source in model_config.keys(), f'Source not available! Choose between: {list(model_config.keys())}'
        return model_config[source]

