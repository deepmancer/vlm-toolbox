import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from config.enums import Metrics
from metric.base import BaseMetricEvaluator
from metric.metric_computers.factory import MetricFactory


class ClassificationMetricEvaluator(BaseMetricEvaluator):
    def __init__(
        self,
        label_handler, 
        main_metric_name=Metrics.ACCURACY,
        complementary_metrics_names=[Metrics.F1, Metrics.RECALL, Metrics.PRECISION, Metrics.ACCURACY, Metrics.G_MEAN, Metrics.BALANCED_ACCURACY],
        top_k=5, 
 ):
        super().__init__()
        self.top_k = top_k
        self.main_metric_name = main_metric_name
        self.complementary_metrics_names = complementary_metrics_names
        self.class_id_label_id_adj_matrix = torch.clone(label_handler.get_class_id_label_id_adj_matrix().int())
        self.class_id_label_id_adj_dict = label_handler.get_mapping_df('class_id', 'label_id', return_type='dict')
        self.top_k = min(self.top_k, len(self.class_id_label_id_adj_dict.keys()))

        self.is_multi_class = torch.any(torch.sum(self.class_id_label_id_adj_matrix == 1, dim=1) > 1).item()
        if not self.is_multi_class:
            self.class_id_label_id_adj_matrix = torch.argmax(self.class_id_label_id_adj_matrix, dim=1).long()
            self.class_id_label_id_adj_dict = {k: v[0] for k, v in self.class_id_label_id_adj_dict.items()}
        
        self.main_metric = MetricFactory.create_metric(metric_name=main_metric_name)

        self.complementary_metrics = MetricFactory.create_metrics(
            include_metric_names=complementary_metrics_names,
            exclude_metric_names=[main_metric_name],
        )
        
        self.initialize()

    def register_metrics(self, metric_names=[]):
        metrics =  MetricFactory.create_metrics(
            include_metric_names=metric_names,
            exclude_metric_names=[self.main_metric_name],
        )
        for metric in metrics:
            if metric not in self.complementary_metrics:
                self.complementary_metrics.append(metric)
        
    def initialize(self):
        columns = ['class_id', 'actual_label_id', 'correct_pred_rank']
        for i in range(1, self.top_k + 1):
            columns.append(f'pred@{i}_label_id')
            columns.append(f'pred@{i}_prob')

        self.column_dtype_mapping = {
            'class_id': 'int32',
            'actual_label_id': 'int32',
            'correct_pred_rank': 'int16',
            **{f'pred@{j+1}_label_id': 'int32' for j in range(self.top_k)},
            **{f'pred@{j+1}_prob': 'float16' for j in range(self.top_k)}
        }
        self.predictions_df = pd.DataFrame(columns=columns).astype(self.column_dtype_mapping)

    def get_main_metric_name(self):
        return self.main_metric_name

    def get_predictions(self):
        return self.predictions_df

    def get_metrics(self, predictions_df=None, main_metric_only=True, top_k=None):
        return self._calculate_overall_metrics(predictions_df=predictions_df, main_metric_only=main_metric_only, top_k=top_k)

    def _compute_metrics(self, true_labels, predicted_labels, main_metric_only=True, classes=None, predicted_probs=None):
        kwargs = dict(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            classes=classes,
            predicted_probs=predicted_probs,
        )
        metrics = {
            self.main_metric_name: self.main_metric.compute(**kwargs)
        }
        if not main_metric_only:
            for metric in self.complementary_metrics:
                metric_value = metric.compute(**kwargs)
                metrics[metric.get_name()] = metric_value

        return metrics

    def _calculate_overall_metrics(self, predictions_df=None, main_metric_only=True, top_k=None):
        predictions_df = self.predictions_df if predictions_df is None else predictions_df
        classes =  np.sort(predictions_df['actual_label_id'].unique())
        top_k = self.top_k if top_k is None else top_k
        pd.DataFrame()
        metrics = []
        for k in range(1, top_k + 1):
            metrics_at_k = dict(top_k=k)
            if k == 1:
                metrics_at_k.update(self._compute_metrics(
                    true_labels=predictions_df['actual_label_id'],
                    predicted_labels=predictions_df['pred@1_label_id'],
                    classes=classes,
                    predicted_probs=predictions_df['pred@1_prob'],
                    main_metric_only=main_metric_only,
                ))
            else:
                top_k = predictions_df['correct_pred_rank'].apply(lambda x: x <= k and x != -1).mean()
                metrics_at_k[f'accuracy'] = top_k
            metrics.append(metrics_at_k)
    
    
        metrics_df = pd.DataFrame(metrics)
        return metrics_df

    def _find_correct_prediction_rank(self, actual_ids, top_k_predicted_ids):
        actual_ids_expanded = np.expand_dims(actual_ids, axis=1)
        correct_predictions = (top_k_predicted_ids == actual_ids_expanded)
        correct_prediction_rank = np.argmax(correct_predictions, axis=1) + 1
        not_found_mask = (correct_predictions.sum(axis=1) == 0)
        correct_prediction_rank[not_found_mask] = -1
        return correct_prediction_rank

    def __call__(self, eval_pred):
        self.initialize()
       
        predictions = eval_pred.predictions
        class_ids = torch.tensor(predictions[0] if isinstance(predictions, tuple) else predictions.m1_ids).long()
        label_ids = torch.tensor(predictions[1] if isinstance(predictions, tuple) else predictions.m2_ids).long()
        unique_label_ids = torch.unique(label_ids)
        logits_per_image = torch.tensor(predictions[4] if isinstance(predictions, tuple) else predictions.m1_m2_logits)
        
        probabilities = F.softmax(logits_per_image, dim=-1)
        topk_values, topk_indices = torch.topk(probabilities, self.top_k, dim=1)
        topk_values = topk_values.numpy()
        predicted_label_ids = unique_label_ids[topk_indices].numpy()
        actual_label_ids = self.class_id_label_id_adj_matrix[class_ids].squeeze().numpy()

        correct_pred_rank = self._find_correct_prediction_rank(actual_label_ids, predicted_label_ids)

        rows = {
            'class_id': class_ids.numpy(),
            'actual_label_id': actual_label_ids,
            'correct_pred_rank': correct_pred_rank
        }

        for j in range(self.top_k):
            rows[f'pred@{j+1}_label_id'] = predicted_label_ids[:, j]
            rows[f'pred@{j+1}_prob'] = topk_values[:, j]
    
        current_batch_predictions = pd.DataFrame(rows).astype(self.column_dtype_mapping)

        self.predictions_df = pd.concat([self.predictions_df, current_batch_predictions], ignore_index=True)

        metrics_df = self._calculate_overall_metrics(predictions_df=current_batch_predictions, main_metric_only=False, top_k=1)
        metrics_dict = metrics_df.iloc[0].to_dict()
        del metrics_dict['top_k']
        return metrics_dict
    
    def __str__(self):
        return f'{self.__class__.__name__}(main_metric={self.main_metric_name}, complementary_metrics={self.complementary_metrics_names}, top_k={self.top_k})'
