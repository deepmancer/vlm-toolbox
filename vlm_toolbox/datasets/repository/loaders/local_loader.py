import os
import torch
import datasets
import dill
import pandas as pd
from typing import Any, Union, Optional, Dict

from config.enums import LocalStorageType
from datasets.repository.base_loader import BaseLoader


class LocalFileLoader(BaseLoader):
    """
    A utility class for loading various types of data from local files.

    This class provides a convenient interface for loading data stored in different
    formats from the local file system. It supports a variety of file types commonly
    used in data science and machine learning workflows, including Pickle, CSV, Parquet,
    Arrow, image folders, JSON, and PyTorch state dictionaries.

    The class uses a class method `load` as the main entry point, which delegates the
    loading process to the appropriate private method based on the specified storage type.
    
    Supported Local Storage Types:
        - PICKLE: Load data from a `.pkl` or `.pickle` file using the `dill` library.
        - CSV: Load data from a `.csv` file into a Pandas DataFrame.
        - PARQUET: Load data from a `.parquet` file into a Pandas DataFrame.
        - ARROW: Load a dataset from Apache Arrow format, often used with Hugging Face datasets.
        - IMAGE_FOLDER: Load a dataset from a directory of images, using the Hugging Face `datasets` library.
        - STATE_DICT: Load a PyTorch state dictionary, typically used for model weights.
        - JSON: Load data from a `.json` file into a Pandas DataFrame.
        - HUGGINGFACE_DATASET: Load a dataset from the Hugging Face datasets library.
    
    Example Usage:
        ```python
        # Load a CSV file into a Pandas DataFrame
        df = LocalFileLoader.load('data/sample.csv', LocalStorageType.CSV)

        # Load a Pickle file
        obj = LocalFileLoader.load('data/object.pkl', LocalStorageType.PICKLE)

        # Load a PyTorch model state dict
        model_state = LocalFileLoader.load('model.pth', LocalStorageType.STATE_DICT)
        ```
    
    Methods:
        load(uri: str, storage_type: Union[str, LocalStorageType], **kwargs: Any) -> Any:
            Loads the data from the specified file path based on the given storage type.
            
        load_async(uri: str, storage_type: Union[str, LocalStorageType], **kwargs: Any) -> Any:
            Asynchronously loads the data from the specified file path based on the given storage type.
    """

    @classmethod
    def load(
        cls,
        uri: str,
        storage_type: Optional[Union[str, LocalStorageType]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Load data from a specified local file based on the provided storage type.

        This method supports loading data from various file formats such as 
        Pickle, CSV, Parquet, Arrow, image folders, JSON, and PyTorch state dictionaries.
        It delegates the loading process to the appropriate helper method based 
        on the `storage_type`.

        Args:
            uri (str): The path to the local file or directory to load.
            storage_type (Optional[Union[str, LocalStorageType]]): The type of the storage format.
                Should be one of the following:
                - LocalStorageType.PICKLE: Load a Pickle file using `dill`.
                - LocalStorageType.CSV: Load a CSV file into a Pandas DataFrame.
                - LocalStorageType.PARQUET: Load a Parquet file into a Pandas DataFrame.
                - LocalStorageType.ARROW: Load a dataset from an Arrow format.
                - LocalStorageType.IMAGE_FOLDER: Load an image dataset from a directory.
                - LocalStorageType.STATE_DICT: Load a PyTorch state dictionary.
                - LocalStorageType.JSON: Load a JSON file into a Pandas DataFrame.
                - LocalStorageType.HUGGINGFACE_DATASET: Load a dataset from the Hugging Face datasets library.

            **kwargs (Any): Additional keyword arguments passed to the specific 
                loading method.

        Returns the loaded data, whose type depends on the file format being loaded.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            ValueError: If the specified `storage_type` is not supported.
            Any: The loaded data, whose type depends on the file format being loaded.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            ValueError: If the specified `storage_type` is not supported.

        Example:
            ```python
            # Load a CSV file into a DataFrame
            df = LocalFileLoader.load('data/sample.csv', LocalStorageType.CSV)

            # Load a PyTorch state dict
            state_dict = LocalFileLoader.load('model.pth', LocalStorageType.STATE_DICT)
            ```
        """
        if not os.path.exists(uri):
            raise FileNotFoundError(f"The file at {uri} does not exist.")

        if storage_type is None:
            storage_type = cls._infer_storage_type_from_extension(uri)

        if storage_type == LocalStorageType.PICKLE:
            return cls._load_pickle(uri, **kwargs)
        elif storage_type == LocalStorageType.CSV:
            return cls._load_csv(uri, **kwargs)
        elif storage_type == LocalStorageType.PARQUET:
            return cls._load_parquet(uri, **kwargs)
        elif storage_type == LocalStorageType.ARROW:
            return cls._load_arrow(uri, **kwargs)
        elif storage_type == LocalStorageType.IMAGE_FOLDER:
            return cls._load_image_folder(uri, **kwargs)
        elif storage_type == LocalStorageType.STATE_DICT:
            return cls._load_state_dict(uri, **kwargs)
        elif storage_type == LocalStorageType.JSON:
            return cls._load_json(uri, **kwargs)
        elif storage_type == LocalStorageType.HUGGINGFACE_DATASET:
            return cls._load_hf_dataset(uri, **kwargs)
        else:
            raise ValueError(f"Data storage type {storage_type} is not supported")
    
    @classmethod
    def _load_pickle(cls, uri: str, **kwargs: Any) -> Any:
        """Load a Pickle file using the dill library."""
        with open(uri, 'rb') as f:
            return dill.load(f, **kwargs)
    
    @classmethod
    def _load_csv(cls, uri: str, **kwargs: Any) -> datasets.Dataset:
        """Load a CSV file into a Hugging Face dataset."""
        pd_dataset = pd.read_csv(uri, **kwargs)
        return datasets.Dataset.from_pandas(pd_dataset)

    @classmethod
    def _load_parquet(cls, uri: str, **kwargs: Any) -> datasets.Dataset:
        """Load a Parquet file into a Hugging Face dataset."""
        pd_dataset = pd.read_parquet(uri, **kwargs)
        return datasets.Dataset.from_pandas(pd_dataset)

    @classmethod
    def _load_arrow(cls, uri: str, **kwargs: Any) -> Union[datasets.DatasetDict, datasets.Dataset]:
        """Load a dataset from Arrow format using Hugging Face datasets."""
        if cls._is_directory(uri):
            data_files = cls._extract_data_files_from_dir(uri, keep_type='arrow')
            return datasets.load_dataset('arrow', data_files=data_files, **kwargs)
        elif cls._is_a_file(uri):            
            return datasets.Dataset.from_file(uri, **kwargs)
        else:
            raise ValueError(f"Unsupported Arrow file format: {uri}")

    @classmethod
    def _load_image_folder(cls, uri: str, **kwargs: Any) -> datasets.Dataset:
        """Load an image dataset from a directory using Hugging Face datasets."""
        if not cls._is_directory(uri):
            raise ValueError(f"Path {uri} is not a directory")
        return datasets.load_dataset('image_folder', data_dir=uri, **kwargs)
    
    @classmethod
    def _load_state_dict(cls, uri: str, version: str = None, **kwargs: Any) -> Any:
        """
        Load a PyTorch state dictionary, optionally with a specific version.
        
        Args:
            uri (str): Path to the state dictionary file.
            version (str, optional): Version of the state dictionary to load.
        
        Returns:
            Any: Loaded PyTorch state dictionary.
        """
        if version:
            version_uri = f"{uri}_v{version}"
            if not os.path.exists(version_uri):
                raise FileNotFoundError(f"The file at {version_uri} does not exist.")
            uri = version_uri
        return torch.load(uri, **kwargs)

    @classmethod
    def _load_json(cls, uri: str, **kwargs: Any) -> datasets.Dataset:
        """Load a JSON file into a Hugging Face dataset."""
        pd_dataset = pd.read_json(uri, **kwargs)
        return datasets.Dataset.from_pandas(pd_dataset)

    @classmethod
    def _load_hf_dataset(cls, uri: str, **kwargs: Any) -> datasets.Dataset:
        """Load a dataset from Hugging Face datasets library."""
        return datasets.load_from_disk(uri, **kwargs)

    @classmethod
    def _is_a_file(cls, uri: str) -> bool:
        """Check if the path is a single file."""
        return os.path.isfile(uri)
    
    @classmethod
    def _is_directory(cls, uri: str) -> bool:
        """Check if the path is a directory."""
        return os.path.isdir(uri)

    @classmethod
    def _infer_storage_type_from_extension(cls, uri: str) -> LocalStorageType:
        """
        Infer the file storage type based on the file extension.
    
        Args:
            uri (str): The path to the file.
    
        Returns:
            LocalStorageType: The inferred storage type based on the file extension.
    
        Raises:
            ValueError: If the file extension is not supported.
        """
        extension = uri.split('.')[-1].lower()
        if extension in ['pkl', 'pickle']:
            return LocalStorageType.PICKLE
        elif extension == 'csv':
            return LocalStorageType.CSV
        elif extension == 'parquet':
            return LocalStorageType.PARQUET
        elif extension == 'arrow':
            return LocalStorageType.ARROW
        elif extension == 'json':
            return LocalStorageType.JSON
        elif extension in ['pth', 'pt', 'ckpt', 'model']:
            return LocalStorageType.STATE_DICT
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
    @classmethod
    def _extract_data_files_from_dir(cls, uri: str, keep_type: str) -> Dict[str, str]:
        """
        Extract data files from a directory.

        Args:
            uri (str): Path to the directory containing data files.
            keep_type (str): The file extension to filter files by.

        Returns:
            Dict[str, str]: A dictionary mapping file names to file paths.
        """
        data_files = {}
        for root, _, file_names in os.walk(uri):
            for file_name in file_names:
                if file_name.endswith(keep_type):
                    prefix = os.path.relpath(root, start=uri).replace(os.path.sep, '.')
                    data_files[prefix] = os.path.join(root, file_name)
        return data_files

__all__ = ["LocalFileLoader"]
