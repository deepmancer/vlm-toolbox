import torch
from typing import ClassVar, Dict

from config.base import BaseConfig
from config.enums import PrecisionDtype


class PrecisionConfig(BaseConfig):
    """
    Manages data precision configurations for training and testing deep-learning models.
    """

    _config: ClassVar[Dict[PrecisionDtype, torch.dtype]] = {
        PrecisionDtype.FP16: torch.float16,
        PrecisionDtype.FP32: torch.float32,
        PrecisionDtype.FP64: torch.float64,
        PrecisionDtype.BF16: torch.bfloat16,
    }

    @classmethod
    def get_dtype(cls, precision_dtype: PrecisionDtype) -> torch.dtype:
        """
        Returns the torch dtype corresponding to the given precision dtype.

        Args:
            precision_dtype (PrecisionDtype): The desired precision data type.

        Returns:
            torch.dtype: The corresponding torch data type.

        Raises:
            ValueError: If the requested precision is not supported or
                        if BF16 is requested but not supported by CUDA.
        """
        dtype = cls.get(precision_dtype)
        if dtype is None:
            raise ValueError(f"Unsupported precision dtype: {precision_dtype}")

        if precision_dtype == PrecisionDtype.BF16 and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA does not support the bf16 dtype.")

        return dtype
