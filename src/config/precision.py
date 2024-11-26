import numpy as np
import torch
from pydantic import Field, ValidationError, field_validator

from config.base import BaseConfig
from config.enums import PrecisionDtype


_PRECISION_CONVERSION_MAP = {
    PrecisionDtype.FP16: {"numpy": np.float16, "torch": torch.float16},
    PrecisionDtype.FP32: {"numpy": np.float32, "torch": torch.float32},
    PrecisionDtype.FP64: {"numpy": np.float64, "torch": torch.float64},
    PrecisionDtype.BF16: {"numpy": np.float32, "torch": torch.bfloat16},
}

class PrecisionConfig(BaseConfig):
    """
    Manages data precision configurations for training and testing deep-learning models.

    Attributes:
        precision_dtype (PrecisionDtype): Specifies the precision data type.
    """
    precision_dtype: PrecisionDtype = Field(
        default=PrecisionDtype.FP32,
        title="Precision Data Type",
        description="Data type for precision control.",
    )

    @field_validator('precision_dtype')
    @classmethod
    def check_precision_compatibility(cls, value: PrecisionDtype) -> PrecisionDtype:
        """
        Validates that the specified precision data type is supported.

        Args:
            value (PrecisionDtype): The desired precision data type.

        Returns:
            PrecisionDtype: The validated precision data type.

        Raises:
            ValueError: If the precision dtype is unsupported or incompatible with the current hardware.
        """
        if value not in _PRECISION_CONVERSION_MAP:
            raise ValidationError(f"Precision data type '{value}' is not supported.")
        
        if value == PrecisionDtype.BF16 and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is not supported on this device.")
        
        return value

    @property
    def numpy_dtype(self) -> np.dtype:
        """
        Retrieves the NumPy data type corresponding to the configured precision.

        Returns:
            np.dtype: The NumPy data type for the configured precision.
        """
        return _PRECISION_CONVERSION_MAP[self.precision_dtype]["numpy"]

    @property
    def torch_dtype(self) -> torch.dtype:
        """
        Retrieves the PyTorch data type corresponding to the configured precision.

        Returns:
            torch.dtype: The PyTorch data type for the configured precision.
        """
        return _PRECISION_CONVERSION_MAP[self.precision_dtype]["torch"]


__all__ = ["PrecisionConfig"]
