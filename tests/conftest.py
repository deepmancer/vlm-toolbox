import os
import sys

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(project_root)

src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

import pytest

from vlm_toolbox.data.data_access.label_handler import LabelHandler


pytest_plugins = [
   "tests.data_fixtures.labels",
]


@pytest.fixture
def label_handler(fake_metadata_df):
    config = {
        'label_column_name': 'class_label',
        'prompt_column': 'prompt',
        'prompt_templates': ['a photo of a {}', 'an image of a {}']
    }
    return LabelHandler(metadata_df=fake_metadata_df, config=config)
