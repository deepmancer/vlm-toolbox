import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(ROOT_DIR)

sys.path.append(ROOT_DIR)

USER_HOME_DIR = os.path.expanduser('~')
ANALYTICS_OUT_DIR = os.path.join(REPO_DIR, 'outputs')
IO_DIR = os.path.join('/home/alireza/', 'io')

EXPERIMENTS_LOGGING_DIR = os.path.join(USER_HOME_DIR, 'vlm_log')
EXPERIMENTS_ROOT_DIR = os.path.join(USER_HOME_DIR, 'io', 'experiments')
EXPERIMENTS_MODEL_DIR = os.path.join(USER_HOME_DIR, 'io', 'model')

ANNOTATIONS_TEMPLATE_PATH = os.path.join(ROOT_DIR, 'annotations')

VISUALIZATIONS_ROOT_DIR = os.path.join(ANALYTICS_OUT_DIR, 'visualization')
SETUPS_DIR = os.path.join(ANALYTICS_OUT_DIR, 'setups')

IMAGE_EMBEDS_TEMPLATE_PATH = IO_DIR + '/{dataset_name}/embedding/{{backbone_name}}/{{source}}/{{split}}/'
IMAGES_TEMPLATE_PATH = IO_DIR + '/{dataset_name}/image/{{split}}/'
