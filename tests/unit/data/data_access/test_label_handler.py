import pytest
import pandas as pd

def test_initialize(label_handler):
    assert label_handler.labels_df is not None
    assert 'label' in label_handler.labels_df.columns
    assert 'label_id' in label_handler.labels_df.columns

def test_get_to_original_class_id_mapping(label_handler):
    mapping = label_handler.get_to_original_class_id_mapping()
    expected_mapping = {0: 0, 1: 1, 2: 2}
    assert mapping == expected_mapping

def test_filter_labels(label_handler):
    filter_dict = {'class_label': ['cat', 'dog']}
    label_handler.filter_labels(filter_dict)
    assert len(label_handler.metadata_df) == 2
    assert label_handler.metadata_df['class_label'].tolist() == ['cat', 'dog']

def test_sync_class_and_label_ids(label_handler):
    label_handler.sync_class_and_label_ids()
    assert all(label_handler.labels_df['class_id'] == label_handler.labels_df['label_id'])

def test_config_prompts(label_handler):
    label_handler.config_prompts()
    assert 'prompt' in label_handler.labels_df.columns

def test_set_label(label_handler):
    label_handler.set_label('class_label')
    assert 'label_id' in label_handler.labels_df.columns

def test_update_label(label_handler):
    label_handler.update_label('class_label')
    assert 'label' in label_handler.labels_df.columns

def test_get_mapping_df(label_handler):
    mapping_df = label_handler.get_mapping_df('class_id', 'label_id')
    assert isinstance(mapping_df, pd.DataFrame)

def test_get_mapping(label_handler):
    mapping = label_handler.get_mapping('class_id', 'label_id')
    assert isinstance(mapping, list)

def test_get_class_id_label_id_adj_matrix(label_handler):
    adj_matrix = label_handler.get_class_id_label_id_adj_matrix()
    assert isinstance(adj_matrix, torch.Tensor)

def test_get_class_id_label_id_mapping(label_handler):
    mapping = label_handler.get_class_id_label_id_mapping()
    assert isinstance(mapping, torch.Tensor)

def test_get_label_id_prompt_id_mapping(label_handler):
    mapping = label_handler.get_label_id_prompt_id_mapping()
    assert isinstance(mapping, torch.Tensor)

def test_get_prompts_df(label_handler):
    prompts_df = label_handler.get_prompts_df()
    assert isinstance(prompts_df, pd.DataFrame)

def test_get_classes_df(label_handler):
    classes_df = label_handler.get_classes_df()
    assert isinstance(classes_df, pd.DataFrame)

def test_get_fine_to_coarse_label_id_mapping(label_handler):
    coarse_mapping = label_handler.get_fine_to_coarse_label_id_mapping()
    assert coarse_mapping is None

def test_get_labels_df(label_handler):
    labels_df = label_handler.get_labels_df()
    assert isinstance(labels_df, pd.DataFrame)

def test_set_prompt_mode(label_handler):
    label_handler.set_prompt_mode(True)
    assert label_handler.is_soft is True

def test_get_num_classes(label_handler):
    num_classes = label_handler.get_num_classes()
    assert num_classes == 3

def test_get_num_labels(label_handler):
    num_labels = label_handler.get_num_labels()
    assert num_labels == 3

def test_get_labels(label_handler):
    labels = label_handler.get_labels()
    assert labels == ['cat', 'dog', 'bird']

def test_get_class_ids(label_handler):
    class_ids = label_handler.get_class_ids()
    assert class_ids == [0, 1, 2]

def test_add_column_to_metadata(label_handler):
    def transform_fn(row, suffix):
        return row['class_label'] + suffix
    
    label_handler.add_column_to_metadata(transform_fn, 'new_label', suffix='_test')
    assert 'new_label' in label_handler.metadata_df.columns

def test_set_prompt_templates(label_handler):
    new_templates = ['a {}']
    label_handler.set_prompt_templates(new_templates)
    assert label_handler.prompt_templates == new_templates

def test_get_metadata_df(label_handler):
    metadata_df = label_handler.get_metadata_df()
    assert isinstance(metadata_df, pd.DataFrame)

def test_get_label_id_column(label_handler):
    column_name = label_handler.get_label_id_column()
    assert column_name == 'label_id'

def test_show(label_handler):
    import io
    output = io.StringIO()
    label_handler.show(logging_fn=output.write)
    assert 'Current label column' in output.getvalue()

@pytest.mark.parametrize("filter_dict, expected_labels", [
    ({'class_label': ['cat']}, ['cat']),
    ({'class_label': ['dog']}, ['dog']),
    ({'class_label': ['cat', 'bird']}, ['cat', 'bird'])
])
def test_filter_labels_parametrized(label_handler, filter_dict, expected_labels):
    label_handler.filter_labels(filter_dict)
    assert label_handler.metadata_df['class_label'].tolist() == expected_labels

def test_set_label_invalid_column(label_handler):
    with pytest.raises(ValueError):
        label_handler.set_label('invalid_label')
