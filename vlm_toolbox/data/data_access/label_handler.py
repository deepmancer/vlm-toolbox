import numpy as np
import pandas as pd
import torch


class LabelHandler:
    def __init__(self, metadata_df, config={}):
        self.metadata_df = metadata_df 
        self.config = config

        self.is_soft = False
        self.soft_prompt_group = None
        self.label_column = self.config.get('label_column_name', 'class_label')
        self.to_original_class_id_mapping = pd.Series(metadata_df['class_id'].values, index=metadata_df['class_id']).to_dict()
        self.label_id_column_name = 'label_id'

        self.initialize()

    def initialize(self):    
        self.hard_prompt_column = self.config.get('prompt_column', 'prompt')
        self.hard_prompt_id_column = self.config.get('prompt_id_column', 'prompt_id')

        self.soft_prompt_column = 'soft_prompt'
        self.soft_prompt_id_column = 'soft_prompt_id'

        self.prompt_templates = self.config['prompt_templates']
        self.default_prompt_template = self.config.get('default_prompt_template', '{}')
        self.context_initialization = self.config.get('context_initialization', None)

        self.labels_df = self.metadata_df.copy()
        self.labels_df['label'] = self.labels_df[self.label_column].copy()
        self.labels_df['label_id'] = self.labels_df['class_id'].copy()
        return self

    @property
    def prompt_id_column(self):
        if self.is_soft:
            return self.soft_prompt_id_column
        return self.hard_prompt_id_column

    @property
    def prompt_column(self):
        if self.is_soft:
            return self.soft_prompt_column
        return self.hard_prompt_column
    
    def get_to_original_class_id_mapping(self):
        return self.to_original_class_id_mapping
    
    def filter_labels(self, filter_dict):
        condition = pd.Series([True] * len(self.metadata_df))
        for col_name, to_keep_values in filter_dict.items():
            condition &= self.metadata_df[col_name].isin(to_keep_values)

        self.metadata_df = (
            self.metadata_df[condition]
            .sort_values(by='class_id')
            .reset_index(drop=True)
        )

        original_class_ids = self.metadata_df['class_id'].copy()
        self.metadata_df['class_id'] = range(len(self.metadata_df))

        self.to_original_class_id_mapping = pd.Series(
            original_class_ids.values,
            index=self.metadata_df['class_id']
        ).to_dict()
        
        self.initialize()
        return self

    def sync_class_and_label_ids(self, column_name='label_id'):
        self.labels_df['class_id'] = self.labels_df[column_name]
        return self

    def config_prompts(self, clean_labels=True, **kwargs):
        if self.is_soft:
            self.config_soft_prompts(**kwargs)
        else:
            self.config_hard_prompts(**kwargs)

        if clean_labels:
            self.labels_df['label'] = self.labels_df['label'].apply(lambda l: l.lower())
            self.labels_df[self.prompt_column] = self.labels_df[self.prompt_column].apply(lambda l: l.lower())
            
        return self

    def config_soft_prompts(self, soft_prompt_group=None, **kwargs):
        soft_prompt_group = soft_prompt_group if soft_prompt_group else self.soft_prompt_group
        if not soft_prompt_group:
            self.labels_df[self.soft_prompt_column] = 'generic'
            self.labels_df[self.soft_prompt_id_column] = 0
        else:
            self.soft_prompt_group = soft_prompt_group  
            self.labels_df[self.soft_prompt_column] = self.labels_df[self.soft_prompt_group]
            self.labels_df[self.soft_prompt_id_column] = pd.factorize(self.labels_df[self.soft_prompt_column])[0]
        return self

    def config_hard_prompts(self, apply_default=True, apply_on_col='label', **kwargs):
        if apply_default:
            prompts = [self.default_prompt_template]
        else:
            prompts = self.prompt_templates
        
        self.labels_df[self.hard_prompt_column] = self.labels_df.apply(
            lambda x: [template.format(x[apply_on_col]) for template in prompts], axis=1
        )

        self.labels_df = (
            self.labels_df.explode(self.hard_prompt_column)
            .reset_index(drop=True)
            .sort_values('class_id', inplace=False)
        )
        self.labels_df[self.hard_prompt_id_column], _ = pd.factorize(self.labels_df[self.hard_prompt_column])
        return self
  
    def set_label(self, label_name='label'):
        if label_name not in self.labels_df.columns:
            raise ValueError(f'{label_name} not in annotations.')
        self.label_column = label_name
        self.labels_df['label_id'] = pd.factorize(self.labels_df[label_name])[0]
        return self
  
    def update_label(self, label_source, flatten=False):
        if label_source is None or label_source == self.label_column:
            return self

        if isinstance(label_source, str):
            self.labels_df['label'] = self.labels_df[label_source]
        elif callable(label_source):
            self.labels_df['label'] = self.labels_df.apply(label_source, axis=1)
        
        self.set_label(label_source)
        if flatten:
            self.labels_df = self.labels_df.explode('label')

        return self

    def get_mapping_df(self, source_col, dest_col, return_type='df'):
        mapping_df = self.labels_df[[source_col, dest_col]].drop_duplicates(keep='first').sort_values(by=source_col).reset_index(drop=True)
        if return_type == 'dict':
            return mapping_df.groupby(source_col)[dest_col].apply(list).to_dict()
        return mapping_df

    def get_mapping(self, source_col, dest_col):
        mapping_df = self.get_mapping_df(source_col, dest_col, return_type='df')
        try:
            return torch.tensor(mapping_df[dest_col].to_numpy(), requires_grad=False).int()
        except Exception:
            return mapping_df[dest_col].to_list()

    def get_class_id_label_id_adj_matrix(self):
        class_label_dict = self.get_mapping_df('class_id', 'label_id', return_type='dict')
        num_classes = self.labels_df['class_id'].nunique()
        num_labels = self.labels_df['label_id'].nunique()
        adj_matrix = np.zeros((num_classes, num_labels), dtype=int)
        for class_id, label_ids in class_label_dict.items():
            adj_matrix[class_id, label_ids] = 1
        
        return torch.tensor(adj_matrix, requires_grad=False).int()

    def get_class_id_label_id_mapping(self):
        return self.get_mapping('class_id', 'label_id')

    def get_label_id_prompt_id_mapping(self):
        mapping_df = self.get_mapping_df('label_id', self.prompt_id_column)
        prompt_ids = mapping_df[self.prompt_id_column].to_numpy()
        return torch.tensor(prompt_ids, requires_grad=False).int()

    def get_prompts_df(self, drop_duplicates=True):
        columns_to_keep = ['label_id', 'label' if self.is_soft else self.prompt_column]
        df = (
            self.labels_df[columns_to_keep]
            .rename(columns={
                self.prompt_column: 'label',
            })
        )
        if drop_duplicates:
            df = df.drop_duplicates(keep='first').sort_values(by='label_id').reset_index(drop=True)

        return df
    
    def get_classes_df(self, drop_duplicates=True):
        columns_to_keep = ['class_id', 'class_label']
        df = (
            self.labels_df[columns_to_keep]
            .rename(columns={
                'class_label': 'label',
            })
        )
        if drop_duplicates:
            df = df.drop_duplicates(keep='first').sort_values(by='class_id').reset_index(drop=True)

        return df

    def get_fine_to_coarse_label_id_mapping(self, coarse_column_name='coarse'):
        if coarse_column_name not in self.labels_df.columns:
            return None
        
        coarse_id_col = f"{coarse_column_name}_id"
        if coarse_id_col not in self.labels_df.columns:
            self.labels_df[coarse_id_col] = pd.factorize(self.labels_df[coarse_column_name])[0]

        mapping_df = self.get_mapping_df('label_id', coarse_id_col)
        coarse_ids = mapping_df[coarse_id_col].to_numpy()
        return torch.tensor(coarse_ids, requires_grad=False).int()
    
    def get_labels_df(self, drop_duplicates=True):
        columns_to_keep = ['label_id', 'label']
        df = self.labels_df[columns_to_keep]
    
        if drop_duplicates:
            df = df.drop_duplicates(keep='first').sort_values(by=columns_to_keep[0]).reset_index(drop=True)
        return df

    def set_prompt_mode(self, is_soft=False):
        self.is_soft = is_soft
        return self

    def get_context_initialization(self):
        return self.context_initialization

    def get_num_classes(self):
        return self.labels_df['class_id'].nunique()

    def get_num_labels(self):
        return self.labels_df['label_id'].nunique()

    def get_labels(self):
        return self.labels_df['label'].drop_duplicates(keep='first').to_list()
    
    def get_class_ids(self):
        mapping = self.get_to_original_class_id_mapping()
        return list(mapping.keys())
    
    def add_column_to_metadata(self, transform_fn, destination_col_name, flatten=False, **kwargs):
        self.metadata_df[destination_col_name] = self.metadata_df.apply(lambda row: transform_fn(row, **kwargs), axis=1)
        if flatten:
            self.metadata_df = self.metadata_df.explode(destination_col_name)

        self.initialize()
        return self

    def set_prompt_templates(self, prompt_templates):
        self.prompt_templates = prompt_templates
        return self

    def get_metadata_df(self):
        return self.metadata_df

    def get_label_id_column(self):
        return self.label_id_column_name
    
    def show(self, logging_fn=print):
        logging_fn(str(self))

    def __str__(self):
        representation_parts = []
        
        representation_parts.append(f'Current label column: {self.label_column}')
        representation_parts.append(f'Number of classes: {self.get_num_classes()}')
        representation_parts.append(f'Number of labels: {self.get_num_labels()}')

        prompt_mode = 'Soft' if self.is_soft else 'Hard'
        representation_parts.append(f'Prompt mode: {prompt_mode}')
        
        if self.is_soft:
            soft_prompt_group = self.soft_prompt_group or 'generic'
            representation_parts.append(f'Soft prompt group: {soft_prompt_group}')
            prompts_cnt = len(self.labels_df[self.soft_prompt_id_column].unique())
            representation_parts.append(f'Number of soft prompts: {prompts_cnt}')
            representation_parts.append(f'Dataset\'s prompt context initialization: {self.context_initialization}')
        else:
            representation_parts.append(f'Hard prompt column: {self.hard_prompt_column}')
            representation_parts.append(f'Default prompt template: {self.default_prompt_template}')
            if 'prompt_templates' in self.config:
                representation_parts.append('Prompt templates:')
                for template in self.config['prompt_templates'][:3]:
                    representation_parts.append(f'  - {template}')
                representation_parts.append('...')

        return '\n'.join(representation_parts)
    
    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items() if key in self.__dict__)
        return f"{class_name}({attributes})"
