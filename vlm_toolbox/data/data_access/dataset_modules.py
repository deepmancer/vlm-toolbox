import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from config.memory import NUM_PROCS
from util.path import mkdir_if_missing
from util.torch_helper import group_operation


class DatasetTransformer:
    @staticmethod
    def aggregate(
        dataset,
        group_by_col_name,
        aggregated_col_name='embedding',
        aggregate_on_col_name='embedding',
        aggregation_method='mean',
    ):
        """
        Aggregate the dataset based on a specified column.

        Args:
            dataset (datasets.Datasest): The input dataset.
            group_by_col_name (str): The name of the column to group by.
            aggregated_col_name (str, optional): The name of the aggregated column. Defaults to 'embedding'.
            aggregate_on_col_name (str, optional): The name of the column to aggregate on. Defaults to 'embedding'.
            aggregation_method (str, optional): The method used for aggregation. Defaults to 'mean'.

        Returns:
            datasets.Datasest: The aggregated dataset.
        """
        values = dataset[aggregate_on_col_name]
        key_ids = dataset[group_by_col_name]

        unique_keys, inverse_keys = torch.unique(torch.tensor(key_ids), return_inverse=True)

        aggregated_values, _ = group_operation(values, inverse_keys, method=aggregation_method, dim=0)

        aggregated_dataset = {
            group_by_col_name: unique_keys.tolist(),
            aggregated_col_name: aggregated_values
        }

        return Dataset.from_dict(aggregated_dataset)

    @staticmethod
    def downsample(dataset, downsample_on_col_name, n_shots, random_state=42):
        """
        Downsamples a dataset based on a specified column.

        Args:
            dataset (datasets.Datasest): The dataset to be downsampled.
            downsample_on_col_name (str): The name of the column to downsample on.
            n_shots (int): The number of samples to keep for each unique value in the specified column.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
           datasets.Datasest: The downsampled dataset.

        """
        value_counts = (
            dataset.select_columns([downsample_on_col_name])
            .to_pandas()
            [downsample_on_col_name]
            .value_counts()
        )
        
        if any(value_counts < n_shots):
            insufficient_classes = value_counts[value_counts < n_shots].index.tolist()
            raise ValueError(f"Insufficient samples for classes: {insufficient_classes}. Each class needs at least {n_shots} samples.")

        identifiers = dataset[downsample_on_col_name]
        indices = list(range(len(dataset)))

        df = pd.DataFrame({'identifier': identifiers, 'index': indices})

        sampled_df = (
            df.groupby('identifier')
            .apply(lambda x: x.sample(n=min(len(x), n_shots), random_state=random_state))
            .reset_index(drop=True)
        )

        downsampled_dataset_indices = sampled_df['index'].to_list()
        downsampled_dataset = dataset.select(downsampled_dataset_indices)
        return downsampled_dataset

    @staticmethod
    def filter_column_wise(dataset, column_name, target_values):
        """
        Filters a Hugging Face dataset based on a specific column and values.

        Args:
            dataset (Dataset): The Hugging Face dataset to filter.
            column_name (str): The name of the column to filter on.
            target_values (list): The values to filter for in the specified column.

        Returns:
            Dataset: The filtered Hugging Face dataset.

        Raises:
            ValueError: If the specified column does not exist in the dataset.
            ValueError: If the dataset type is not supported.
        """
        if column_name not in dataset.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset")

        column_values = dataset[column_name]
        if not isinstance(column_values[0], str):
            column_values = torch.tensor(column_values)
        else:
            column_values, _ = pd.factorize(column_values)
            target_values = pd.factorize(target_values)[0]

        if isinstance(column_values, np.ndarray):
            is_index_eligible = np.isin(column_values, np.array(target_values))
        elif isinstance(column_values, torch.Tensor):
            is_index_eligible = torch.isin(column_values, torch.tensor(target_values)).numpy()
        elif isinstance(column_values, list):
            is_index_eligible = np.array([value in target_values for value in column_values])
        else:
            raise ValueError("Unsupported dataset type")

        eligible_indices = np.where(is_index_eligible)[0]
        return dataset.select(eligible_indices)

class DatasetIOHandler:
    @staticmethod
    def save_to_disk(dataset, directory):
        mkdir_if_missing(directory)
        dataset.save_to_disk(directory)

    @staticmethod
    def load_from_disk(directory, is_imagefolder=False):
        if is_imagefolder:
            dataset = load_dataset('imagefolder', data_dir=directory)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[list(dataset.keys())[0]]
        else:
            dataset = load_from_disk(directory)
        return dataset
    
    @staticmethod
    def load_dataset(path, dataset_type, splits=None):
        if dataset_type == 'disk':
            return DatasetIOHandler.load_from_disk(path, is_imagefolder=False)
        elif dataset_type == 'imagefolder':
            return DatasetIOHandler.load_from_disk(path, is_imagefolder=True)
            # return dataset[list(dataset.keys())[0]]
        elif dataset_type == 'huggingface':
            return load_dataset(path, split=splits, num_proc=NUM_PROCS)
        else:
            raise NotImplementedError(f"Dataset type '{dataset_type}' not recognized.")
