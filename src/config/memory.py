import os

import datasets
import torch

in_memory_max_gigabytes = 30
in_memory_max_bytes = in_memory_max_gigabytes * 1024 ** 3

datasets.config.IN_MEMORY_MAX_SIZE = in_memory_max_bytes
os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = f'{in_memory_max_bytes}'
torch.backends.cudnn.benchmark = True

NUM_PROCS = 3
NUM_SHARDS = 4