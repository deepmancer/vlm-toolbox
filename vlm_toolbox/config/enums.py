from enum import Enum
from typing import List

class BaseEnum(str, Enum):
    @classmethod
    def get(cls, name_str: str) -> str:
        try:
            return cls[name_str.upper()].value
        except KeyError:
            raise ValueError(f"{name_str} is not a valid name for {cls.__name__}")

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]


class Datasets(BaseEnum):
    IMAGENET_1K = 'imagenet1k'
    FOOD101 = 'food101'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    STANFORD_CARS = 'stanford_cars'
    INATURALIST = 'inaturalist2021'
    MSCOCO_CAPTIONS = 'mscoco_captions'

class CLIPBackbones(BaseEnum):
    VIT_B_32 = 'clip_vit_b_32'
    VIT_B_16 = 'clip_vit_b_16'
    VIT_L_14 = 'clip_vit_l_14'
    RESNET_50 = 'clip_resnet_50'
    RESNET_101 = 'clip_resnet_101'
    RESNET_50_4 = 'clip_resnet_50_4'
    RESNET_50_16 = 'clip_resnet_50_16'
    RESNET_50_64 = 'clip_resnet_50_64'
    VIT_L_14_336PX = 'clip_vit_l_14_336px'

class VisionLanguageBackbones(BaseEnum):
    CLIP = 'clip'

class VisionBackbones(BaseEnum):
    DYNO_V2_GIANT = 'dyno_v2_giant'
    
class LanguageBackbones(BaseEnum):
    ALL_MINILM_L6_V2 = 'all_minilm_l6_v2'
    ALL_MPNET_BASE_V2 = 'all_mpnet_base_v2'


class SamplingType(BaseEnum):
    OVER_SAMPLING = 'over_sampling'
    UNDER_SAMPLING = 'under_sampling'
    HYBRID = 'hybrid'


class SamplingStrategy(BaseEnum):
    RANDOM_OVER_SAMPLING = 'random_over_sampling'
    BORDERLINE_SMOTE = 'borderline_smote'
    SMOTE = 'smote'
    SVM_SMOTE = 'svm_smote'
    ADASYN = 'adasyn'
    KMEANS_SMOTE = 'kmeans_smote'
    RANDOM_UNDER_SAMPLING = 'random_under_sampling'
    EDITED_NEAREST_NEIGHBOURS = 'edited_nearest_neighbours'
    CONDENSED_NEAREST_NEIGHBOUR = 'condensed_nearest_neighbour'
    NEAR_MISS = 'near_miss'
    SMOTEENN = 'smoteenn'
    SMOTETOMEK = 'smotetomek'
    CLUSTER_CENTROIDS = 'cluster_centroids'
    ALL_KNN = 'all_knn'
    NEIGHBOURHOOD_CLEANING_RULE = 'neighbourhood_cleaning_rule'
    ONE_SIDED_SELECTION = 'one_sided_selection'
    TOMEK_LINKS = 'tomek_links'

class AnnotationsProviders(BaseEnum):
    DATASET = 'dataset'
    REPOSITORY = 'repository'
    MANUAL = 'manual'

class DataTypes(BaseEnum):
    IMAGE = 'image'
    TEXT = 'text'
    EMBEDDING = 'embedding'

class ModelBackboneTypes(BaseEnum):
    VISION = 'vision'
    LANGUAGE = 'language'
    VISION_LANGUAGE = 'vision_language'

class Trainers(BaseEnum):
    CLIP = 'clip'
    COOP = 'coop'

class Stages(BaseEnum):
    TRAIN = 'train'
    EVAL = 'validation'
    PREPROCESS = 'preprocess'

class ModelProviders(BaseEnum):
    HUGGINGFACE = 'huggingface'
    OPEN_AI = 'open_ai'
    TIMM = 'timm'
    MANUAL = 'manual'
    
class DatasetProviders(BaseEnum):
    HUGGINGFACE = 'huggingface'
    TORCHVISION = 'torchvision'
    KAGGLE = 'kaggle'
    MANUAL = 'manual'

class LossType(BaseEnum):
    CONTRASTIVE_LOSS = 'contrastive'
    LABEL_SMOOTHING_LOSS = 'label_smoothing'
    WEIGHTED_L2_LOSS = 'weighted_l2'
    WEIGHTED_L1_LOSS = 'weighted_l1'
    MARGIN_METRIC_LOSS = 'margin_metric'
    ENLARGED_LARGE_MARGIN_LOSS = 'enlarged_large_margin_loss'


class DataStatus(BaseEnum):
    RAW = 'raw'
    PREPROCESSED = 'preprocessed'
    EMBEDDING = 'embedding'

class ModalityType(BaseEnum):
    IMAGE = 'image'
    TEXT = 'text'


class Modalities(BaseEnum):
    M1 = 'm1'
    M2 = 'm2'

class StorageType(BaseEnum):
    HF_DATASET = 'hf_dataset'
    IMAGE_FOLDER = 'image_folder'
    STATE_DICT = 'state_dict'
    PARQUET = 'parquet'
    PICKLE = 'pickle'
    ARROW = 'arrow'
    JSON = 'json'
    CSV = 'csv'


class Setups(BaseEnum):
    FULL = 'full'
    TRAIN_ONLY = 'train_only'
    EVAL_ONLY = 'eval_only'


class ModelType(BaseEnum):
    PRETRAINED = 'pretrained'
    ZERO_SHOT = 'zero_shot'
    FEW_SHOT = 'few_shot'
    FULL_TRAINED = 'full_trained'


class PrecisionDtypes(BaseEnum):
    FP16 = 'fp16'
    BF16 = 'bf16'
    FP32 = 'fp32'
    FP64 = 'fp64'


class Metrics(BaseEnum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1'
    COHEN_KAPPA = 'cohen_kappa'
    M_CORR_COEFF = 'm_corr_coeff'
    BALANCED_ACCURACY = 'balanced_accuracy'
    BALANCED_ACCURACY_WEIGHTED = 'balanced_accuracy_weighted'
    G_MEAN = 'g_mean'
    AUC_ROC = 'auc_roc'
    SENSITIVITY = 'sensitivity'
    SPECIFICITY = 'specificity'
