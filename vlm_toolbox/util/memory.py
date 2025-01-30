import gc
import os
import signal

import psutil
import torch
from pynvml import *


def print_gpu_utilization(logging_fn=print):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logging_fn(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result, logging_fn=print):
    logging_fn(f"Time: {result.metrics['train_runtime']:.2f}")
    logging_fn(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    for proc in psutil.process_iter(attrs=['pid', 'status']):
        if proc.info['status'] == psutil.STATUS_ZOMBIE:
            try:
                os.kill(proc.info['pid'], signal.SIGKILL)
            except (PermissionError, ProcessLookupError):
                pass
