from pathlib import Path
import sys
from typing import Union


class PathConfig:
    """
    A configuration class for managing project directory paths.
    Automatically determines and constructs paths relative to the repository structure.
    Allows modification of output-related paths, while source, repository, and home directories remain fixed.
    """

    OUTPUTS_DIR = 'outputs'
    IO_DIR = 'io'
    DATASETS_DIR = 'datasets'
    EXPERIMENTS_DIR = 'experiments'
    MODELS_DIR = 'models'
    VISUALIZATION_DIR = 'visualization'
    LOGS_DIR = 'logs'
    SETUPS_DIR = 'setups'
    ANNOTATIONS_DIR = 'annotations'

    def __init__(self):
        """Initializes paths based on the project structure."""
        self._src_dir: Path = self._get_src_dir()
        self._repo_dir: Path = self._get_repo_dir()
        self._user_home_dir: Path = Path.home()

        # Initialize paths with default values
        self._analytics_out_dir: Path = self._repo_dir / self.OUTPUTS_DIR
        self._experiments_logging_dir: Path = self._analytics_out_dir / self.LOGS_DIR
        self._datasets_dir: Path = self._repo_dir / self.IO_DIR / self.DATASETS_DIR
        self._experiments_root_dir: Path = self._user_home_dir / self.IO_DIR / self.EXPERIMENTS_DIR
        self._experiments_model_dir: Path = self._user_home_dir / self.IO_DIR / self.MODELS_DIR
        self._annotations_template_path: Path = self._src_dir / self.ANNOTATIONS_DIR
        self._visualizations_root_dir: Path = self._analytics_out_dir / self.VISUALIZATION_DIR
        self._setups_dir: Path = self._analytics_out_dir / self.SETUPS_DIR

    @staticmethod
    def _get_src_dir() -> Path:
        """Determines the source directory based on the location of the current file."""
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def _get_repo_dir() -> Path:
        """Determines the repository root directory based on the source directory."""
        return PathConfig._get_src_dir().parent

    def add_src_to_sys_path(self) -> None:
        """Adds the source directory to the Python system path for imports."""
        src_dir_str = str(self._src_dir)
        if src_dir_str not in sys.path:
            sys.path.append(src_dir_str)

    @property
    def src_dir(self) -> Path:
        """Returns the source directory."""
        return self._src_dir

    @property
    def repo_dir(self) -> Path:
        """Returns the repository root directory."""
        return self._repo_dir

    @property
    def user_home_dir(self) -> Path:
        """Returns the user's home directory."""
        return self._user_home_dir

    @property
    def analytics_out_dir(self) -> Path:
        """Returns the analytics output directory."""
        return self._analytics_out_dir

    @analytics_out_dir.setter
    def analytics_out_dir(self, path: Union[str, Path]) -> None:
        """Updates the analytics output directory."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._analytics_out_dir = path

    @property
    def experiments_logging_dir(self) -> Path:
        """Returns the experiments logging directory."""
        return self._experiments_logging_dir

    @experiments_logging_dir.setter
    def experiments_logging_dir(self, path: Union[str, Path]) -> None:
        """Updates the experiments logging directory."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._experiments_logging_dir = path

    @property
    def datasets_dir(self) -> Path:
        """Returns the datasets directory."""
        return self._datasets_dir

    @datasets_dir.setter
    def datasets_dir(self, path: Union[str, Path]) -> None:
        """Updates the datasets directory."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._datasets_dir = path

    @property
    def experiments_root_dir(self) -> Path:
        """Returns the root directory for experiments."""
        return self._experiments_root_dir

    @experiments_root_dir.setter
    def experiments_root_dir(self, path: Union[str, Path]) -> None:
        """Updates the root directory for experiments."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._experiments_root_dir = path

    @property
    def experiments_model_dir(self) -> Path:
        """Returns the directory for experiment models."""
        return self._experiments_model_dir

    @experiments_model_dir.setter
    def experiments_model_dir(self, path: Union[str, Path]) -> None:
        """Updates the directory for experiment models."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._experiments_model_dir = path

    @property
    def annotations_template_path(self) -> Path:
        """Returns the annotations template directory."""
        return self._annotations_template_path

    @annotations_template_path.setter
    def annotations_template_path(self, path: Union[str, Path]) -> None:
        """Updates the annotations template directory."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._annotations_template_path = path

    @property
    def visualizations_root_dir(self) -> Path:
        """Returns the root directory for visualizations."""
        return self._visualizations_root_dir

    @visualizations_root_dir.setter
    def visualizations_root_dir(self, path: Union[str, Path]) -> None:
        """Updates the root directory for visualizations."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._visualizations_root_dir = path

    @property
    def setups_dir(self) -> Path:
        """Returns the setups directory."""
        return self._setups_dir

    @setups_dir.setter
    def setups_dir(self, path: Union[str, Path]) -> None:
        """Updates the setups directory."""
        path = Path(path)
        self._validate_absolute_path(path)
        self._setups_dir = path

    @staticmethod
    def _validate_absolute_path(path: Union[str, Path]) -> None:
        """Validates that the provided path is an absolute path."""
        if not Path(path).is_absolute():
            raise ValueError("Path must be an absolute path.")

__all__ = ['PathConfig']
