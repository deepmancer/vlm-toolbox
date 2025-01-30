from config.base import BaseConfig
from config.enums import SamplingStrategy, SamplingType


class SamplingConfig(BaseConfig):
    config = {
        SamplingType.OVER_SAMPLING: {
            SamplingStrategy.RANDOM_OVER_SAMPLING: {
                'sampling_strategy': 'auto',
            },
            SamplingStrategy.BORDERLINE_SMOTE: {
                'sampling_strategy': 'auto',
                'kind': 'borderline-1',
                'k_neighbors': 5,
                'm_neighbors': 10,
            },
            SamplingStrategy.SMOTE: {
                'sampling_strategy': 'auto',
                'k_neighbors': 5,
            },
            SamplingStrategy.SVM_SMOTE: {
                'sampling_strategy': 'auto',
                'k_neighbors': 5,
                'm_neighbors': 10,
            },
            SamplingStrategy.ADASYN: {
                'sampling_strategy': 'auto',
                'n_neighbors': 5,
            },
            SamplingStrategy.KMEANS_SMOTE: {
                'sampling_strategy': 'auto',
                'k_neighbors': 5,
            },
        },
        SamplingType.UNDER_SAMPLING: {
            SamplingStrategy.RANDOM_UNDER_SAMPLING: {
                'sampling_strategy': 'auto',
            },
            SamplingStrategy.EDITED_NEAREST_NEIGHBOURS: {
                'sampling_strategy': 'auto',
                'n_neighbors': 3,
            },
            SamplingStrategy.CONDENSED_NEAREST_NEIGHBOUR: {
                'sampling_strategy': 'auto',
                'n_neighbors': 1,
            },
            SamplingStrategy.NEAR_MISS: {
                'sampling_strategy': 'auto',
                'n_neighbors': 1,
                'version': 1,
                'n_neighbors_ver3': 3,
            },
            SamplingStrategy.CLUSTER_CENTROIDS: {
                'sampling_strategy': 'auto',
                'voting': 'auto',
            },
            SamplingStrategy.ALL_KNN: {
                'sampling_strategy': 'auto',
                'n_neighbors': 3,
                'kind_sel': 'all',
                'allow_minority': False,
            },
            SamplingStrategy.NEIGHBOURHOOD_CLEANING_RULE: {
                'sampling_strategy': 'auto',
                'n_neighbors': 3,
                'kind_sel': 'deprecated',
                'threshold_cleaning': 0.5,
            },
            SamplingStrategy.ONE_SIDED_SELECTION: {
                'sampling_strategy': 'auto',
                'n_seeds_S': 1,
            },
            SamplingStrategy.TOMEK_LINKS: {
                'sampling_strategy': 'auto',
            },
        },
        SamplingType.HYBRID: {
            SamplingStrategy.SMOTEENN: {
                'sampling_strategy': 'auto',
            },
            SamplingStrategy.SMOTETOMEK: {
                'sampling_strategy': 'auto',
            },
        },
    }

    @staticmethod
    def get_config(sampling_type, sampling_strategy, **kwargs):
        type_config = SamplingConfig.config.get(sampling_type)
        if not type_config:
            raise ValueError(f"No configuration found for sampling type: {sampling_type}")

        strategy_config = type_config.get(sampling_strategy)
        if not strategy_config:
            raise ValueError(f"No configuration found for sampling strategy: {sampling_strategy}")

        default_kwargs = {k: v for k, v in strategy_config.items() if v is not None}
        default_kwargs.update(kwargs)

        return default_kwargs
