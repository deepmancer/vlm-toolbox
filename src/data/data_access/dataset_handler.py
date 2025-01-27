import random

import datasets
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional
from datasets import Dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

from config.enums import DataStatus, ModalityType, Stages
from data.data_access.dataset_modules import (
    DatasetIOHandler,
    DatasetTransformer,
)
from data.sample.factory import SamplerFactory


class SingleModalDatasetHandler:
    def __init__(self, dataset, modality, config={}):
        """
        Initializes a new instance of the SingleModalDatasetHandler class.

        Parameters:
            dataset (Dataset): The dataset object.
            modality (Modality): The modality object.
            config (dict, optional): Additional configuration options (default: {}).

        Returns:
            None
        """
        self.dataset = dataset
        self.modality = modality
        self.config = config
        self.identifier = modality.get_identifier()
        self.index_column = self.identifier
        self.index = None
        self.is_few_shot = False
        self.n_shots = None

    @property
    def return_type(self):
        return 'pt' if (self.is_embedded() or self.is_preprocessed()) else None

    @property
    def values(self):
        return self.modality.get_current_values()

    def get_necessary_cols(self):
        return list(set([self.get_identifier()] + self.get_values()))

    def get_modality(self):
        return self.modality

    def get_dataset_features(self):
        num_unique_ids = self.get_unique_count()
        return f"{self.dataset.__repr__()}\n" + f"num unique ids: {num_unique_ids}"

    def post_init(self, persist=False, logging_fn=print):
        if self.get_modality().get_type() == ModalityType.IMAGE and self.get_modality().is_raw():
            features = self.dataset.features[self.get_values()[0]]
            current_mode = features.mode
            if current_mode != 'RGB':
                logging_fn('Converting Images to RGB')
                features.mode = 'RGB'
                self.dataset = self.dataset.cast_column(self.get_values()[0], features)

        if persist:
            self.persist()

        return self

    def get_dataset(
        self,
        return_pt=None,
        keep_necessary_cols_only=False,
        prototypical=False,
        id_only=False,
        split_size=None,
        random_state=None,
        sampling_type=None,
        sampling_strategy=None,
        sampling_column=None,
        sampling_kwargs={},
    ):
        dataset = self.dataset
        
        if id_only:
            to_return_cols = [self.get_identifier()]
        elif keep_necessary_cols_only:
            to_return_cols = self.get_necessary_cols()
        else:
            to_return_cols = dataset.column_names

        if return_pt is not None:
            dataset = dataset.with_format('torch')
        else:
            dataset = dataset.with_format(self.return_type)

        if split_size is not None:
            dataset = dataset.train_test_split(test_size=split_size, seed=random_state)
            dataset = {
                Stages.TRAIN: dataset['train'],
                Stages.EVAL: dataset['test'],
            }
            if sampling_type:
                dataset[Stages.TRAIN] = self.get_sampled_dataset(
                    sampling_type,
                    sampling_strategy,
                    dataset=dataset[Stages.TRAIN],
                    identifier=sampling_column,
                    **sampling_kwargs,
                )
            if prototypical:
                for split in dataset.keys():
                    dataset[split] = DatasetTransformer.aggregate(
                        dataset[split], self.identifier, embed_col, embed_col,
                    )
        else:
            if sampling_type:
                dataset = self.get_sampled_dataset(
                    sampling_type,
                    sampling_strategy,
                    dataset=dataset,
                    identifier=sampling_column,
                    **sampling_kwargs,
                )
            if prototypical:
                embed_col = self.modality.get_embedding_keys(output=True)[0]
                dataset = DatasetTransformer.transformer_handler.aggregate(
                    dataset, self.identifier, embed_col, embed_col,
                )

        if isinstance(dataset, dict):
            for split in dataset.keys():
                dataset[split] = dataset[split].select_columns(to_return_cols)
        else:
            dataset = dataset.select_columns(to_return_cols)

        return dataset

    def build_index(self, column_name=None):
        column_name = column_name if column_name else self.index_column
        if column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")
        df = pd.DataFrame({
            'index_column': self.dataset[column_name],
            'idx': range(len(self.dataset))
        })
        grouped_df = df.groupby('index_column')['idx'].apply(list)
        self.index = grouped_df.to_dict()
        self.index_column = column_name
        return self

    def remove_columns(self, columns):
        self.dataset = self.dataset.remove_columns(columns)
        return self

    def assign_column(
        self,
        values,
        col_name=None,
        to_drop_col_names=[],
        col_data_status=None,
        **kwargs,
    ):
        if not col_name:
            col_name = self.modality.preprocessed_keys[0] if col_data_status == DataStatus.EMBEDDING else self.modality.embedding_key

        if col_name in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns([col_name])
            
        new_dataset = datasets.Dataset.from_dict({col_name: values})
        self.dataset = datasets.concatenate_datasets([self.dataset, new_dataset], axis=1)

        if to_drop_col_names:
             self.dataset = self.dataset.remove_columns(to_drop_col_names)

        self.post_jobs(data_status=col_data_status)
        return self

    def to_embedding(
        self,
        transform_fn,
        keep_necessary_cols_only=False,
        batched=True,
        col_data_status=DataStatus.EMBEDDING,
        batch_size=1024,
        optimized=False,
        dataset: Optional[Dataset] = None,
        num_proc=None,
        keep_in_memory=True,
        **kwargs,
    ):
        dataset = dataset if dataset is not None else self.dataset
        source_col_names = kwargs.pop('source_col_names', self.get_values())

        if optimized:
            assert self.is_preprocessed() is False and self.is_embedded() is False
            assert len(source_col_names) == 1
            value = source_col_names[0]

            df = dataset.to_pandas().reset_index(drop=False).rename(columns={'index': 'original_idx'})
            unique_values_df = df.groupby([value])['original_idx'].apply(np.array).reset_index()
            unique_dataset = Dataset.from_pandas(unique_values_df)
            
            unique_dataset = self.apply_transformation(
                transform_fn, source_col_names=source_col_names, keep_source_cols=True, 
                keep_in_memory=keep_in_memory, num_proc=num_proc, batched=batched, batch_size=batch_size,
                dataset=unique_dataset, **kwargs,
            )
            dataset_df = (
                unique_dataset.to_pandas()
                .explode('original_idx')
                .sort_values(by=['original_idx'])
                .drop(columns=['original_idx'])
                .reset_index(drop=True)
            )
            dataset_df[self.identifier] = dataset[self.identifier]
            dataset = Dataset.from_pandas(dataset_df)
            self.dataset = dataset
        else:
            self.dataset = self.apply_transformation(
                transform_fn, source_col_names=source_col_names,
                keep_source_cols=True, num_proc=num_proc, keep_in_memory=keep_in_memory,
                batched=batched, batch_size=batch_size, dataset=dataset, **kwargs,
            )

        self.post_jobs(data_status=col_data_status)

        if keep_necessary_cols_only:
            necessary_cols = self.get_necessary_cols()
            for col in self.dataset.column_names:
                if col not in necessary_cols:
                    self.dataset = self.dataset.remove_columns([col])

        return self

    def to_prototypical_representation(
        self,
        group_by_col_name=None,
        aggregate_on_col_name=None,
        destination_col_name=None,
        dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        True if dataset is None else False
        dataset = dataset if dataset is not None else self.dataset
        
        group_by_col_name = group_by_col_name or self.identifier
        aggregate_on_col_name = aggregate_on_col_name or self.modality.get_embedding_keys(output=True)[0]
        destination_col_name = destination_col_name or self.modality.get_embedding_keys(output=True)[0]

        dataset.set_format(type='torch', columns=[aggregate_on_col_name, group_by_col_name])

        self.dataset = DatasetTransformer.aggregate(
            dataset, group_by_col_name, aggregate_on_col_name, destination_col_name,
        )

        self.post_jobs(reindex=True)
        return self
    
    def filter(
        self,
        values,
        column=None,
        in_place=False,
        random_state=None,
        return_pt=None,
        keep_necessary_cols_only=False,
        dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        disable_progress_bar()
        random.seed(random_state)
        
        dataset = dataset if dataset is not None else self.dataset
        column = column if column else self.identifier
        filtered_dataset = DatasetTransformer.filter_column_wise(dataset, column, values)
        if keep_necessary_cols_only:
            necessary_cols = self.get_necessary_cols()
            filtered_dataset = filtered_dataset.select_columns(necessary_cols)

        if return_pt is not None:
            filtered_dataset = filtered_dataset.with_format('pt')
        else:
            filtered_dataset = filtered_dataset.with_format(self.return_type)

        enable_progress_bar()
        if in_place:
            self.dataset = filtered_dataset
            self.post_jobs(reindex=True)
            return self
        else:
            return filtered_dataset

    def to_few_shot_dataset(
        self,
        n_shots,
        downsample_on_col_name=None,
        random_state=42,
        dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        dataset = dataset if dataset is not None else self.dataset
        downsample_on_col_name = downsample_on_col_name if downsample_on_col_name else self.identifier

        self.dataset = DatasetTransformer.downsample(dataset, downsample_on_col_name, n_shots, random_state)
        self.post_jobs(reindex=True, n_shots=n_shots)
        return self

    def apply_transformation(
        self,
        transform_fn,
        source_col_names=None,
        keep_source_cols=False,
        batched=True,
        batch_size=1024,
        in_place=False,
        dataset: Optional[Dataset] = None,
        num_proc=None,
        keep_in_memory=False,
        **kwargs,
    ):
        dataset = dataset if dataset is not None else self.dataset
        transformed_dataset = dataset.map(
            transform_fn, batched=batched, input_columns=source_col_names, num_proc=num_proc, keep_in_memory=keep_in_memory, 
            batch_size=batch_size, remove_columns=source_col_names or [] if not keep_source_cols else [], **kwargs,
        )
        
        if in_place:
            self.dataset = transformed_dataset
            return self
        else:
            return transformed_dataset

    def map_dataset_identifier(
        self,
        mapping,
        dataset: Optional[Dataset] = None,
        store_in_col_name: Optional[str] = None,
        **kwargs: Any,
    ):
        dataset = dataset if dataset is not None else self.dataset
        old_identifiers = dataset[self.identifier]
        if isinstance(old_identifiers, np.ndarray):
            old_identifiers = torch.tensor(old_identifiers)
        elif isinstance(old_identifiers, list):
            old_identifiers = torch.tensor(old_identifiers)
        
        new_identifiers = mapping[old_identifiers]

        if store_in_col_name is not None:
            self.dataset = dataset.add_column(store_in_col_name, new_identifiers.tolist())
        else:
            self.dataset = dataset.remove_columns(self.identifier).add_column(self.identifier, new_identifiers.tolist())
        self.post_jobs(reindex=True if store_in_col_name is None else False)
        return self

    def shuffle(self, random_state=42):
        self.dataset = self.dataset.shuffle(seed=random_state)
        self.post_jobs(reindex=True)
        return self

    def get_unique_count(self, col=None):
        col = col if col else self.identifier
        return len(self.dataset.unique(col))

    def get_unique(self, col=None):
        col = col if col else self.identifier
        return list(self.dataset.unique(col))

    def is_preprocessed(self):
        return self.modality.is_preprocessed()

    def is_embedded(self):
        return self.modality.is_embedded()

    def set_few_shot_flag(self, n_shots):
        self.is_few_shot = True
        self.n_shots = n_shots

    def reset_few_shot_flag(self):
        self.is_few_shot = False
        self.n_shots = None

    def set_identifier(self, identifier):
        assert identifier in self.dataset.column_names
        self.identifier = identifier
        self.post_jobs(reindex=True)
        return self

    def get_values(self):
        return self.modality.get_current_values()

    def get_identifier(self):
        return self.identifier

    def get_config(self):
        return self.config

    def get_type(self):
        return self.modality.get_type()

    def save_to_disk(self, directory):
        dataset = self.get_dataset()
        DatasetIOHandler.save_to_disk(dataset, directory)
        return self

    def load_from_disk(self, directory, status=DataStatus.EMBEDDING, persist=False):
        self.dataset = load_from_disk(directory, keep_in_memory=False)
        self.post_jobs(reset=True, data_status=status)

        if persist:
            self.persist()

        return self

    def post_jobs(self, reset=False, reset_flags=False, data_status=None, reindex=False, n_shots=None, **kwargs):
        if reset:
            if data_status:
                self.modality.update_status(data_status)
            else:
                self.modality.reset_status()
                
            self.reset_few_shot_flag()
            self.build_index()
            return

        if reset_flags:
            self.reset_few_shot_flag()

        if data_status:
            self.modality.update_status(data_status)
            if data_status != DataStatus.RAW:
                self.dataset.set_format(type='torch', columns=self.get_values())

        if reindex:
            if self.index is not None:
                self.build_index()

        if n_shots:
            self.set_few_shot_flag(n_shots)

        return

    def show(self, minimal=True, logging_fn=print):
        if minimal:
            logging_fn(str(self))
        else:
            logging_fn(self.detailed_str())

    def persist(self):
        self.dataset = self.dataset.map(lambda x, idx: x, with_indices=True, keep_in_memory=True)
        return self

    def get_sampled_dataset(
        self,
        sampling_type,
        sampling_strategy,
        dataset: Optional[Dataset] = None,
        identifier=None,
        value=None,
        **kwargs,
    ):
        dataset = dataset if dataset is not None else self.dataset
        group_sampling = False
        if identifier is not None and identifier != self.identifier:
            if identifier not in dataset.column_names:
                raise ValueError(f"Column '{identifier}' not found in the dataset.")
            group_sampling = True   
        else:
            identifier = self.get_identifier()

        value = value or self.get_values()[0]
        
        y = np.array(dataset[self.identifier]).astype(int)
        X = np.array(dataset[value])
        
        sampler = SamplerFactory.create_sampler(sampling_type, sampling_strategy, **kwargs)
        if group_sampling:
            y_group = np.array(dataset[identifier]).astype(int)
            X_resampled, y_resampled = sampler.fit_resample(X, y, y_group=y_group)
        else:
            X_resampled, y_resampled = sampler.fit_resample(X, y)

        resampled_data = {
            self.identifier: y_resampled,
            value: X_resampled,
        }
        resampled_dataset = datasets.Dataset.from_dict(resampled_data)
        return resampled_dataset.with_format(self.return_type)

    def get_class_count(self, identifier=None):
        identifier = identifier or self.get_identifier()
        _, counts = np.unique(self.dataset.select_columns([identifier]).with_format(None)[identifier], return_counts=True)
        if self.return_type == 'pt':
            counts = torch.tensor(counts)
        else:
            counts = np.array(counts)  
        return counts

    def detailed_str(self):
        representation_parts = [
            str(self),
            '',
            f"Features: {self.dataset.features}"
        ]

        if self.config:
            representation_parts.append('')
            representation_parts.append('Dataset Config:')
            for key, value in self.config.items():
                if isinstance(value, (list, dict)):
                    iterable = value.items() if isinstance(value, dict) else value
                    limited_items = list(iterable)[:5]
                    item_representation = ', '.join(f'{item}' for item in limited_items)
                    representation_parts.append(f'-    {key}: [{item_representation}{", ..." if len(iterable) > 5 else ""}]')
                else:
                    representation_parts.append(f'-    {key}: {value}')
                    
        return '\n'.join(representation_parts)

    def __str__(self):
        representation_parts = [
            str(self.dataset),
            '',
            f'Identifier column name: {self.get_identifier()}',
            f'Value column names: {self.get_values()}',
            '',
            str(self.modality)
        ]
        return '\n'.join(representation_parts)

    def __repr__(self):    
        class_name = self.__class__.__name__
        attributes = {
            'dataset': self.dataset,
            'modality': self.modality,
            'config': self.config,
            'identifier': self.get_identifier(),
            'values': self.get_values(),
        }
        attributes_str = ', '.join(f'{key}={value!r}' for key, value in attributes.items())
        return f"{class_name}({attributes_str})"
