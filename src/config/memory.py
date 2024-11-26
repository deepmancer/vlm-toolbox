import os
import datasets
import torch
from pydantic import Field
from config.base import BaseConfig


class MemoryConfig(BaseConfig):
    """
    Manages memory configuration for datasets, environment variables, and PyTorch.

    Provides a unified interface for configuring memory-related parameters globally,
    ensuring consistency and efficiency across various libraries.
    """

    max_gigabytes: float = Field(
        default=30.0,
        title="Maximum Memory in Gigabytes",
        description=(
            "Maximum allowed memory in gigabytes for datasets and other operations."
        ),
        ge=0.0,
    )
    num_procs: int = Field(
        default=1,
        title="Number of Processes",
        description="Number of processes to use for datasets library.",
        ge=1,
    )
    num_shards: int = Field(
        default=1,
        title="Number of Shards",
        description="Number of shards to use for data loading.",
        ge=1,
    )

    @property
    def max_bytes(self) -> int:
        """
        Converts max_gigabytes to bytes for internal configuration use.

        Returns:
            int: Maximum memory in bytes.
        """
        return int(self.max_gigabytes * (1024 ** 3))

    def configure(self) -> None:
        """
        Apply memory configurations globally across libraries.
        """
        self._configure_datasets()
        self._configure_environment()
        self._configure_torch()

    def _configure_datasets(self) -> None:
        """
        Configure memory settings for the `datasets` library.
        """
        datasets.config.IN_MEMORY_MAX_SIZE = self.max_bytes

    def _configure_environment(self) -> None:
        """
        Set environment variables for memory-related configuration.

        Sets HF_DATASETS_IN_MEMORY_MAX_SIZE for compatibility with the Hugging Face datasets library.
        """
        os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = str(self.max_bytes)

    def _configure_torch(self) -> None:
        """
        Configure PyTorch for optimal performance based on the memory configuration.

        Enables the CuDNN benchmark for improved performance on fixed input sizes.
        """
        torch.backends.cudnn.benchmark = True


__all__ = ["MemoryConfig"]
