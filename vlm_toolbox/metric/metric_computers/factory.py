from config.enums import Metrics
from metric.metric_computers.classification import (
    Accuracy,
    BalancedAccuracy,
    BalancedAccuracyWeighted,
    CohenKappa,
    F1Score,
    GMean,
    MatthewsCorrCoef,
    Precision,
    Recall,
    Sensitivity,
    Specificity,
)


class MetricFactory:
    metrics = {
        Metrics.PRECISION: Precision,
        Metrics.RECALL: Recall,
        Metrics.F1: F1Score,
        Metrics.COHEN_KAPPA: CohenKappa,
        Metrics.M_CORR_COEFF: MatthewsCorrCoef,
        Metrics.BALANCED_ACCURACY: BalancedAccuracy,
        Metrics.G_MEAN: GMean,
        Metrics.BALANCED_ACCURACY_WEIGHTED: BalancedAccuracyWeighted,
        Metrics.ACCURACY: Accuracy,
        Metrics.SENSITIVITY: Sensitivity,
        Metrics.SPECIFICITY: Specificity,
    }

    @staticmethod
    def create_metric(metric_name):
        if metric_name in MetricFactory.metrics:
            return MetricFactory.metrics[metric_name]()
        else:
            raise ValueError(f"Metric '{metric_name}' is not supported")

    @staticmethod
    def create_metrics(exclude_metric_names=[], include_metric_names=[]):
        if not include_metric_names:
            include_metric_names = set(MetricFactory.metrics.keys())
        else:
            include_metric_names = set(include_metric_names)
    
        exclude_metric_names = set(exclude_metric_names)
    
        return [
            metric() for name, metric in MetricFactory.metrics.items()
            if name in include_metric_names and name not in exclude_metric_names
        ]
