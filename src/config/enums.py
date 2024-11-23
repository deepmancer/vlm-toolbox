from enum import Enum
from typing import List, Dict

enum_registry: Dict[str, Enum] = {}


def register_enum(name: str):
    """Decorator to register enums with a given name."""
    def decorator(enum_cls):
        enum_registry[name] = enum_cls
        return enum_cls
    return decorator


class BaseEnum(str, Enum):
    """Base class for enums with helper methods."""

    @classmethod
    def get(cls, name: str):
        """Get the enum member by its name."""
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"'{name}' is not a valid name in {cls.__name__}")

    @classmethod
    def values(cls) -> List[str]:
        """Get a list of all values in the enum."""
        return [member.value for member in cls]

    @classmethod
    def names(cls) -> List[str]:
        """Get a list of all member names in the enum."""
        return [member.name for member in cls]

    @classmethod
    def from_value(cls, value: str):
        """Get the enum member by its value."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"'{value}' is not a valid value in {cls.__name__}")


@register_enum('device_type')
class DeviceType(BaseEnum):
    CPU = 'cpu'
    CUDA = 'cuda'


@register_enum('dataset_name')
class DatasetName(BaseEnum):
    IMAGENET_1K = 'imagenet1k'
    FOOD101 = 'food101'
    CIFAR100 = 'cifar100'
    STANFORD_CARS = 'stanford_cars'
    INATURALIST = 'inaturalist2021'
    MSCOCO_CAPTIONS = 'mscoco_captions'


@register_enum('backbone_name')
class BackboneName(BaseEnum):
    CLIP_VIT_B_32 = 'vit_b_32'
    CLIP_VIT_B_16 = 'vit_b_16'
    CLIP_VIT_L_14 = 'vit_l_14'
    CLIP_RESNET_50 = 'resnet_50'
    CLIP_RESNET_101 = 'resnet_101'
    CLIP_RESNET_50_4 = 'resnet_50_4'
    CLIP_RESNET_50_16 = 'resnet_50_16'
    CLIP_RESNET_50_64 = 'resnet_50_64'
    CLIP_VIT_L_14_336PX = 'vit_l_14_336px'


@register_enum('trainer_name')
class TrainerName(BaseEnum):
    CLIP = 'clip'
    COOP = 'coop'


@register_enum('stage')
class Stage(BaseEnum):
    TRAIN = 'train'
    VALIDATION = 'validation'


@register_enum('split')
class Split(BaseEnum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


@register_enum('provider')
class Provider(BaseEnum):
    OPEN_AI = 'open_ai'
    HUGGINGFACE = 'huggingface'


@register_enum('loss_wrapper')
class LossWrapper(BaseEnum):
    GROUP_MEAN_AGGREGATOR = 'group_mean_aggregator'


@register_enum('sampling_type')
class SamplingType(BaseEnum):
    OVER_SAMPLING = 'over_sampling'
    UNDER_SAMPLING = 'under_sampling'
    HYBRID = 'hybrid'


@register_enum('sampling_strategy')
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


@register_enum('loss_type')
class LossType(BaseEnum):
    CONTRASTIVE_LOSS = 'contrastive'
    LABEL_SMOOTHING_LOSS = 'label_smoothing'
    WEIGHTED_L2_LOSS = 'weighted_l2'
    WEIGHTED_L1_LOSS = 'weighted_l1'
    MARGIN_METRIC_LOSS = 'margin_metric'
    ENLARGED_LARGE_MARGIN_LOSS = 'enlarged_large_margin'


@register_enum('prompting_type')
class PromptingType(BaseEnum):
    HARD = 'hard'
    SOFT = 'soft'


@register_enum('data_stage')
class DataStage(BaseEnum):
    RAW = 'raw'
    PREPROCESSED = 'preprocessed'
    EMBEDDING = 'embedding'


@register_enum('modality_type')
class ModalityType(BaseEnum):
    IMAGE = 'image'
    TEXT = 'text'


@register_enum('modality_identifier')
class ModalityIdentifier(BaseEnum):
    IMAGE = 'class_id'
    TEXT = 'label_id'


@register_enum('annotations_columns')
class AnnotationsColumns(BaseEnum):
    IDENTIFIER = 'class_id'
    LABEL = 'class_label'


@register_enum('modality_index')
class ModalityIndex(BaseEnum):
    M1 = 'm1'
    M2 = 'm2'


@register_enum('data_source_type')
class DataSourceType(BaseEnum):
    USER_CREATED = 'user_created'
    LIBRARY = 'library'


@register_enum('user_data_storage_type')
class UserDataStorageType(BaseEnum):
    IMAGE_FOLDER = 'image_folder'
    DISK = 'disk'
    SERIALIZED = 'serialized'


@register_enum('dataset_loader_backend')
class DatasetLoaderBackend(BaseEnum):
    TORCH = 'torch'
    HUGGINGFACE = 'huggingface'
    POLARS = 'polars'


@register_enum('setup_type')
class SetupType(BaseEnum):
    FULL = 'full'
    TRAIN_ONLY = 'train_only'
    EVAL_ONLY = 'eval_only'


@register_enum('model_type')
class ModelType(BaseEnum):
    PRETRAINED = 'pretrained'
    ZERO_SHOT = 'zero_shot'
    FEW_SHOT = 'few_shot'
    FULL_TRAINED = 'full_trained'


@register_enum('precision_dtype')
class PrecisionDtype(BaseEnum):
    FP16 = 'fp16'
    BF16 = 'bf16'
    FP32 = 'fp32'
    FP64 = 'fp64'


@register_enum('metric_name')
class MetricName(BaseEnum):
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


@register_enum('lr_scheduler')
class LRScheduler(BaseEnum):
    COSINE = 'cosine'
    TANH = 'tanh'
    STEP = 'step'
    MULTISTEP = 'multistep'
    PLATEAU = 'plateau'
    POLY = 'poly'


@register_enum('optimizer')
class Optimizer(BaseEnum):
    SGD = 'sgd'
    NESTEROV = 'nesterov'
    MOMENTUM = 'momentum'
    SGDP = 'sgdp'
    SGDW = 'sgdw'
    NESTEROVW = 'nesterovw'
    ADAM = 'adam'
    ADAMW = 'adamw'
    ADAMP = 'adamp'
    NADAM = 'nadam'
    NADAMW = 'nadamw'
    RADAM = 'radam'
    ADAMAX = 'adamax'
    ADABELIEF = 'adabelief'
    RADABELIEF = 'radabelief'
    ADADELTA = 'adadelta'
    ADAGRAD = 'adagrad'
    ADAFACTOR = 'adafactor'
    ADANP = 'adanp'
    ADANW = 'adanw'
    LAMB = 'lamb'
    LAMBC = 'lambc'
    LARC = 'larc'
    LARS = 'lars'
    NLARC = 'nlarc'
    NLARS = 'nlars'
    MADGRAD = 'madgrad'
    MADGRADW = 'madgradw'
    NOVOGRAD = 'novograd'
    NVNOVOGRAD = 'nvnovograd'
    RMSPROP = 'rmsprop'
    RMSPROPTF = 'rmsproptf'
    LION = 'lion'
    ADAHESSIAN = 'adahessian'
    FUSEDSGD = 'fusedsgd'
    FUSEDMOMENTUM = 'fusedmomentum'
    FUSEDADAM = 'fusedadam'
    FUSEDADAMW = 'fusedadamw'
    FUSEDLAMB = 'fusedlamb'
    FUSEDNOVOGRAD = 'fusednovograd'
    BNBSGD = 'bnbsgd'
    BNBSGD8BIT = 'bnbsgd8bit'
    BNBMOMENTUM = 'bnbmomentum'
    BNBMOMENTUM8BIT = 'bnbmomentum8bit'
    BNBADAM = 'bnbadam'
    BNBADAM8BIT = 'bnbadam8bit'
    BNBADAMW = 'bnbadamw'
    BNBADAMW8BIT = 'bnbadamw8bit'
    BNBLAMB = 'bnblamb'
    BNBLAMB8BIT = 'bnblamb8bit'
    BNBLARS = 'bnblars'
    BNBLARS8BIT = 'bnblars8bit'
    BNBLION = 'bnblion'
    BNBLION8BIT = 'bnblion8bit'
