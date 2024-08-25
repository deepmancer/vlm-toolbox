import torchvision.datasets as datasets
from typing import Any

from datasets.repository.base_loader import BaseLoader

class TorchVisionLoader(BaseLoader):
    @classmethod
    def load(cls, uri: str, **kwargs: Any) -> Any:
        """
        Load a dataset from the TorchVision datasets library.

        Args:
            uri (str): The identifier of the dataset to load.
            **kwargs (Any): Additional keyword arguments passed to the dataset constructor.

        Returns:
            Any: The loaded dataset.
        """
        dataset_class = getattr(datasets, uri)
        return dataset_class(**kwargs)

__all__ = ["TorchVisionLoader"]
