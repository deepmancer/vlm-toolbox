import os

import datasets
import torch

in_memory_max_bytes = 30064771072
datasets.config.IN_MEMORY_MAX_SIZE = in_memory_max_bytes
os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = f'{in_memory_max_bytes}'
torch.backends.cudnn.benchmark = True

NUM_PROCS = 3
NUM_SHARDS = 4
