from datasets import load_dataset
from typing import Any

from datasets.repository.base_loader import BaseLoader

class HuggingFaceDatasetsLoader(BaseLoader):
    @classmethod
    def load(cls, uri: str, **kwargs: Any) -> Any:
        """
        Load a dataset from the Hugging Face datasets library.

        Args:
            uri (str): The identifier of the dataset to load.
            **kwargs (Any): Additional keyword arguments passed to `datasets.load_dataset`.

        Returns:
            Any: The loaded dataset.
        """
        return load_dataset(uri, **kwargs)

__all__ = ["HuggingFaceDatasetsLoader"]
