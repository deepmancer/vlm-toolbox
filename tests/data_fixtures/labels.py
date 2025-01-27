import pandas as pd
import pytest

def generate_fake_labels_df():
    data = {
        'class_id': list(range(20)),
        'class_label': ['cat', 'dog', 'bird', 'car', 'truck', 'bus', 'tree', 'flower', 'grass', 'mountain', 'lake', 'river', 'ocean', 'cloud', 'sun', 'moon', 'star', 'ant', 'mosquito', 'worm'],
        'coarse': ['animal', 'animal', 'animal', 'vehicle', 'vehicle', 'vehicle', 'vegetation', 'vegetation', 'vegetation', 'landscape', 'landscape', 'landscape', 'landscape', 'sky', 'sky', 'sky', 'sky', 'insect', 'insect', 'insect'],
        'supercategory': ['living being', 'living being', 'living being', 'man-made object', 'man-made object', 'man-made object', 'nature', 'nature', 'nature', 'nature', 'nature', 'nature', 'nature', 'nature', 'nature', 'nature', 'nature', 'living being', 'living being', 'living being'],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def fake_metadata_df():
    return generate_fake_labels_df()