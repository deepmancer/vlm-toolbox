from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)

from config.enums import SamplingStrategy, SamplingType
from config.sampler import SamplingConfig
from data.sample.group_under_sampling_wrapper import GroupUnderSamplingWrapper

class SamplerFactory:
    sampler_mapping = {
        SamplingStrategy.RANDOM_OVER_SAMPLING: RandomOverSampler,
        SamplingStrategy.BORDERLINE_SMOTE: BorderlineSMOTE,
        SamplingStrategy.SMOTE: SMOTE,
        SamplingStrategy.SVM_SMOTE: SVMSMOTE,
        SamplingStrategy.ADASYN: ADASYN,
        SamplingStrategy.KMEANS_SMOTE: KMeansSMOTE,
        SamplingStrategy.RANDOM_UNDER_SAMPLING: RandomUnderSampler,
        SamplingStrategy.EDITED_NEAREST_NEIGHBOURS: EditedNearestNeighbours,
        SamplingStrategy.CONDENSED_NEAREST_NEIGHBOUR: CondensedNearestNeighbour,
        SamplingStrategy.NEAR_MISS: NearMiss,
        SamplingStrategy.SMOTEENN: SMOTEENN,
        SamplingStrategy.SMOTETOMEK: SMOTETomek,
        SamplingStrategy.CLUSTER_CENTROIDS: ClusterCentroids,
        SamplingStrategy.ALL_KNN: AllKNN,
        SamplingStrategy.NEIGHBOURHOOD_CLEANING_RULE: NeighbourhoodCleaningRule,
        SamplingStrategy.ONE_SIDED_SELECTION: OneSidedSelection,
        SamplingStrategy.TOMEK_LINKS: TomekLinks,
    }

    @staticmethod
    def create_sampler(sampling_type, sampling_strategy=None, **kwargs):
        if sampling_strategy is None:
            if sampling_type == SamplingType.OVER_SAMPLING:
                sampling_strategy = SamplingStrategy.RANDOM_OVER_SAMPLING
            elif sampling_type == SamplingType.UNDER_SAMPLING:
                sampling_strategy = SamplingStrategy.RANDOM_UNDER_SAMPLING
            else:
                raise ValueError

        strategy_config = SamplingConfig.get_config(sampling_type, sampling_strategy, **kwargs)
        sampler_class = SamplerFactory.sampler_mapping.get(sampling_strategy)
        if sampler_class is None:
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")
        
        sampler = sampler_class(**strategy_config)
        if sampling_type == SamplingType.UNDER_SAMPLING:
            return GroupUnderSamplingWrapper(sampler)

        return sampler
