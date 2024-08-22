from pydantic import BaseModel, Field, field_validator
from decouple import config
from typing import Optional

from config.enums import PrecisionDtypes

class GeneralSetting(BaseModel):
    dataset_cache_max_memory_bytes: Optional[int] = Field(
        default_factory=lambda: config('DATASET_CACHE_MAX_MEMORY_BYTES', cast=int),
        description="Maximum memory allowed for dataset caching in bytes"
    )
    torch_cudnn_benchmark_enabled: Optional[bool] = Field(
        default_factory=lambda: config('TORCH_CUDNN_BENCHMARK_ENABLED', cast=bool),
        description="Enables benchmarking mode in cuDNN (PyTorch configuration)"
    )
    multiprocessing_process_count: Optional[int] = Field(
        default_factory=lambda: config('MULTIPROCESSING_PROCESS_COUNT', cast=int),
        description="Number of processes to be used in multiprocessing"
    )
    log_level: Optional[str] = Field(
        default_factory=lambda: config('LOG_LEVEL', default='INFO'),
        description="Logging level for the application (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    dataset_processing_precision: Optional[str] = Field(
        default_factory=lambda: config('DATASET_PROCESSING_PRECISION', default=PrecisionDtypes.FP32),
        description="Precision setting for the data processing stage (e.g., fp16, bf16, fp32, fp64)"
    )
    training_precision: Optional[str] = Field(
        default_factory=lambda: config('TRAINING_PRECISION', default=PrecisionDtypes.FP32),
        description="Precision setting for the model training stage (e.g., fp16, bf16, fp32, fp64)"
    )
    validation_precision: Optional[str] = Field(
        default_factory=lambda: config('VALIDATION_PRECISION', default=PrecisionDtypes.FP32),
        description="Precision setting for the model validation stage (e.g., fp16, bf16, fp32, fp64)"
    )

    @field_validator('training_precision', 'validation_precision', mode='before')
    def validate_precision(cls, value):
        if value not in PrecisionDtypes.__dict__.values():
            raise ValueError(f"Unsupported precision dtype: {value}")
        return value

__all__ = ['GeneralSetting']