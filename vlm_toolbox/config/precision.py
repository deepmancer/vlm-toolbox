import torch

from config.base import BaseConfig
from config.enums import PrecisionDtypes


class PrecisionConfig(BaseConfig):
    config = {
        PrecisionDtypes.FP16: torch.float16,
        PrecisionDtypes.FP32: torch.float32,
        PrecisionDtypes.FP64: torch.float64,
        PrecisionDtypes.BF16: torch.bfloat16,
    }
    @staticmethod
    def get_config(precision_dtype):
        dtype = PrecisionConfig.get(precision_dtype)
        if precision_dtype == PrecisionDtypes.BF16:
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"Cuda does not support the bf16 dtype.") 
        return dtype
