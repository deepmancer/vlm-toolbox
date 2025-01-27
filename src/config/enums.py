from enum import Enum, EnumMeta


class BaseEnumMeta(EnumMeta):
    def __getattribute__(cls, name):
        member = super().__getattribute__(name)
        if isinstance(member, Enum):
            return member.value
        return member


class BaseEnum(Enum, metaclass=BaseEnumMeta):
    @classmethod
    def get(cls, name_str):
        try:
            return cls[name_str.upper()].value
        except KeyError:
            raise ValueError(f"{name_str} is not a valid name for {cls.__name__}")

    @classmethod
    def get_values(cls):
        return [member.value for member in cls]


class ImageDatasets(BaseEnum):
    IMAGENET_1K = 'imagenet1k'
    FOOD101 = 'food101'
    CIFAR100 = 'cifar100'
    STANFORD_CARS = 'stanford_cars'
    INATURALIST = 'inaturalist2021'
    MSCOCO_CAPTIONS = 'mscoco_captions'


class ImageBackbones(BaseEnum):
    DYNO_V2_GIANT = 'dyno_v2_giant'


class TextBackbones(BaseEnum):
    ALL_MPNET_BASE_V2 = 'all_mpnet_base_v2'
    ALL_MINILM_L6_V2 = 'all_minilm_l6_v2'


class CLIPBackbones(BaseEnum):
    CLIP_VIT_B_32 = 'vit_b_32'
    CLIP_VIT_B_16 = 'vit_b_16'
    CLIP_VIT_L_14 = 'vit_l_14'
    CLIP_RESNET_50 = 'resnet_50'
    CLIP_RESNET_101 = 'resnet_101'
    CLIP_RESNET_50_4 = 'resnet_50_4'
    CLIP_RESNET_50_16 = 'resnet_50_16'
    CLIP_RESNET_50_64 = 'resnet_50_64'
    CLIP_VIT_L_14_336PX = 'vit_l_14_336px'


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


class Backbones(BaseEnum):
    IMAGE = 'image'
    TEXT = 'text'
    MULTIMODAL = 'multimodal'


class Trainers(BaseEnum):
    CLIP = 'clip'
    COOP = 'coop'

class Stages(BaseEnum):
    TRAIN = 'train'
    EVAL = 'validation'
    PREPROCESS = 'preprocess'


class Sources(BaseEnum):
    OPEN_AI = 'open_ai'
    HUGGINGFACE = 'huggingface'


class LossWrappers(BaseEnum):
    COARSELY_SUPERVISED_LOSS = 'coarsely_supervised'


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
    DISK = 'disk'
    IMAGE_FOLDER = 'imagefolder'
    HUGGING_FACE = 'huggingface'


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
