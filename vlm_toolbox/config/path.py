import os
import sys

# Set the SRC_DIR to the directory containing the 'vlm_toolbox' directory
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(SRC_DIR)

# Add SRC_DIR to the system path
sys.path.append(SRC_DIR)

# Define the new directory paths based on the updated structure
IO_DIR = os.path.join(REPO_DIR, 'io')

# Experiment directories
EXPERIMENTS_ROOT_DIR = os.path.join(IO_DIR, 'experiments')
EXPERIMENTS_LOGGING_DIR = os.path.join(EXPERIMENTS_ROOT_DIR, 'logs')
EXPERIMENTS_MODEL_DIR = os.path.join(EXPERIMENTS_ROOT_DIR, 'models')
EXPERIMENTS_RESULTS_DIR = os.path.join(EXPERIMENTS_ROOT_DIR, 'results')
EXPERIMENTS_VISUALIZATIONS_DIR = os.path.join(EXPERIMENTS_ROOT_DIR, 'visualizations')

# Models directory
MODELS_DIR = os.path.join(IO_DIR, 'models')

# Annotations directory
ANNOTATIONS_PATH = os.path.join(IO_DIR, 'annotations')

# Dataset directory
DATASETS_DIR = os.path.join(IO_DIR, 'datasets')
