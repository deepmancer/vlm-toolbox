import os

import numpy as np
import pandas as pd

from config.metric import MetricIOConfig


class BaseMetricEvaluator:
    def get_metrics(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_main_metric_name(self):
        raise NotImplementedError
    
    def get_metrics(self, predictions_df=None, main_metric_only=True, **kwargs):
        raise NotImplementedError

    def get_predictions(self, *args, **kwargs):
        raise NotImplementedError
        
    def is_greater_better(self):
        return True
    
    def save(self, setup, save_predictions=True, directory=None):
        to_save_dfs = dict(overall=self.get_metrics(main_metric_only=False))
        if save_predictions:
            to_save_dfs.update(dict(per_sample=self.get_predictions()))
      
        to_save_dir = directory if directory is not None else MetricIOConfig.get_config(setup)
        
        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        def convert_float16_to_float32(df):
            for col in df.columns:
                if df[col].dtype == np.float16:
                    df[col] = df[col].astype(np.float32)
            return df

        saved_paths = {}
        for name, df in to_save_dfs.items():
            df = convert_float16_to_float32(df)  # Apply the conversion
            file_path = to_save_dir + name + '.parquet'
            df['trainer_name'] = setup.get_trainer_name()
            df.to_parquet(file_path)
            saved_paths[name] = file_path

        return saved_paths

    @classmethod
    def load(cls, setup, directory=None):
        saved_dir = directory if directory is not None else MetricIOConfig.get_config(setup)
        metrics = {}
        if os.path.exists(saved_dir):
            for filename in os.listdir(saved_dir):
                file_path = os.path.join(saved_dir, filename)
                if filename.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    raise FileNotFoundError(f"No directory found for {saved_dir}")
                metric_scope = os.path.splitext(filename)[0]
                metrics[metric_scope] = df

        if 'overall' in metrics:
            metrics['overall'] = metrics['overall'][metrics['overall']['top_k'] <= setup.get_top_k()]
        return metrics

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items() if key in self.__dict__)
        return f"{class_name}({attributes})"
