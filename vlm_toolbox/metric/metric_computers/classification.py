import numpy as np
from imblearn.metrics import (
    classification_report_imbalanced,
    geometric_mean_score,
    make_index_balanced_accuracy,
    sensitivity_score,
    specificity_score,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config.enums import Metrics
from metric.metric_computers.base import BaseMetric


class ClassificationReportImbalanced:
    def compute(
        self,
        true_labels,
        predicted_labels,
        labels=None,
        target_names=None,
        sample_weight=None,
        digits=2,
        alpha=0.1,
        zero_division='warn',
        per_class=False,
        **kwargs        
    ):
        report = classification_report_imbalanced(
            true_labels, predicted_labels, labels=labels, target_names=target_names, 
            sample_weight=sample_weight, digits=digits, alpha=alpha, 
            output_dict=True, zero_division=zero_division
        )

        renamed_report = {
            'per_class': {},
            'overall': {}
        }

        metric_mapping = {
            'pre': Metrics.PRECISION,
            'rec': Metrics.RECALL,
            'spe': Metrics.SPECIFICITY,
            'f1': Metrics.F1,
            'geo': Metrics.G_MEAN,
            'iba': Metrics.BALANCED_ACCURACY,
            'sup': 'support',
        }

        for key, metrics in report.items():
            if key.startswith('avg_'):
                metric_name = key.replace('avg_', '')
                renamed_report['overall'][metric_mapping[metric_name]] = metrics
            elif key == 'total_support':
                renamed_report['overall']['total_support'] = metrics
            else:
                renamed_report['per_class'][key] = {
                    metric_mapping[metric]: value for metric, value in metrics.items()
                }

        if not per_class:
            return renamed_report['overall']
        
        return renamed_report['per_class']

class Precision(BaseMetric):
    name = Metrics.PRECISION

    def compute(self, true_labels, predicted_labels, average='macro', **kwargs):
        return precision_score(true_labels, predicted_labels, average=average, zero_division=0)

class Sensitivity(BaseMetric):
    name = Metrics.SENSITIVITY

    def compute(self, true_labels, predicted_labels, average='macro', **kwargs):
        return sensitivity_score(true_labels, predicted_labels, average=average)

class Specificity(BaseMetric):
    name = Metrics.SPECIFICITY

    def compute(self, true_labels, predicted_labels, average='macro', **kwargs):
        return specificity_score(true_labels, predicted_labels, average=average)

class Recall(BaseMetric):
    name = Metrics.RECALL

    def compute(self, true_labels, predicted_labels, average='macro', **kwargs):
        return recall_score(true_labels, predicted_labels, average=average, zero_division=0)

class F1Score(BaseMetric):
    name = Metrics.F1

    def compute(self, true_labels, predicted_labels, average='macro', **kwargs):
        return f1_score(true_labels, predicted_labels, average=average, zero_division=0)

class CohenKappa(BaseMetric):
    name = Metrics.COHEN_KAPPA

    def compute(self, true_labels, predicted_labels, **kwargs):
        return cohen_kappa_score(true_labels, predicted_labels)

class MatthewsCorrCoef(BaseMetric):
    name = Metrics.M_CORR_COEFF

    def compute(self, true_labels, predicted_labels, **kwargs):
        return matthews_corrcoef(true_labels, predicted_labels)

class BalancedAccuracy(BaseMetric):
    name = Metrics.BALANCED_ACCURACY

    def compute(self, true_labels, predicted_labels, **kwargs):
        return balanced_accuracy_score(true_labels, predicted_labels)

class GMean(BaseMetric):
    name = Metrics.G_MEAN

    def compute(self, true_labels, predicted_labels, average='weighted', **kwargs):
        return geometric_mean_score(true_labels, predicted_labels, average=average)

class BalancedAccuracyWeighted(BaseMetric):
    name = Metrics.BALANCED_ACCURACY_WEIGHTED

    def compute(self, true_labels, predicted_labels, **kwargs):
        cm = confusion_matrix(true_labels, predicted_labels)
        unique_classes = np.unique(true_labels)
        class_weights = np.bincount(true_labels) / len(true_labels)
        weighted_sum = np.sum([class_weights[i] * (cm[i, i] / np.sum(cm[i]) if np.sum(cm[i]) != 0 else 0) for i in unique_classes])
        return weighted_sum

class AUCROC(BaseMetric):
    name = Metrics.AUC_ROC

    def compute(self, true_labels, predicted_labels, predicted_probs, classes, **kwargs):
        true_labels_binarized = label_binarize(true_labels, classes=classes)
        return roc_auc_score(true_labels_binarized, predicted_probs, average='macro', multi_class='ovr')

class Accuracy(BaseMetric):
    name = Metrics.ACCURACY

    def compute(self, true_labels, predicted_labels, **kwargs):
        return accuracy_score(true_labels, predicted_labels)
