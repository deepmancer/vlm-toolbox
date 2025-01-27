import ast

import pandas as pd

from config.annotations import AnnotationsConfig
from data.data_access.label_handler import LabelHandler


class LabelHandleFactory:
    @staticmethod
    def create_from_config(dataset_config):
        def convert_string_to_list(s):
            try:
                if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
                    return ast.literal_eval(s)
                return s
            except (ValueError, SyntaxError):
                return s

        identifier = dataset_config.get('identifier')
        label_name = dataset_config.get('label_column_name')

        metadata_df = (
            pd.read_csv(dataset_config['annotations_path'])
            .sort_values(identifier, inplace=False)
            .rename(columns={identifier: 'class_id'})
            .reset_index(drop=True)
        )
        for column in metadata_df.columns:
            if metadata_df[column].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
                metadata_df[column] = metadata_df[column].apply(convert_string_to_list)

        metadata_df['class_label'] = metadata_df[label_name]
       
        return LabelHandler(metadata_df, config=dataset_config)
    
    @staticmethod
    def create(dataset_name):
        dataset_config = AnnotationsConfig.get_config(dataset_name)
        return LabelHandleFactory.create_from_config(dataset_config)
        
