from pathlib import Path
import sys
from pydantic import Field, model_validator, ValidationError
from config.base import BaseConfig


class PathConfig(BaseConfig):
    """
    Configuration class for managing project directory paths.

    Inherits from BaseConfig to integrate with the broader configuration framework.
    Automatically determines and constructs paths relative to the repository structure,
    allowing modification of output-related paths while keeping core paths fixed.
    """

    outputs_dir: Path = Field(
        default_factory=lambda: PathConfig._default_outputs_dir(),
        title="Outputs Directory",
        description="Base directory for all output-related files.",
    )
    io_dir: Path = Field(
        default_factory=lambda: PathConfig._default_io_dir(),
        title="IO Directory",
        description="Directory for input-output operations.",
    )
    datasets_dir: Path = Field(
        default_factory=lambda: PathConfig._default_datasets_dir(),
        title="Datasets Directory",
        description="Directory for datasets storage.",
    )
    experiments_root_dir: Path = Field(
        default_factory=lambda: PathConfig._default_experiments_root_dir(),
        title="Experiments Root Directory",
        description="Root directory for storing experiment data.",
    )
    models_dir: Path = Field(
        default_factory=lambda: PathConfig._default_models_dir(),
        title="Models Directory",
        description="Directory for storing models.",
    )
    visualizations_dir: Path = Field(
        default_factory=lambda: PathConfig._default_visualizations_dir(),
        title="Visualizations Directory",
        description="Directory for visualization outputs.",
    )
    logs_dir: Path = Field(
        default_factory=lambda: PathConfig._default_logs_dir(),
        title="Logs Directory",
        description="Directory for storing logs.",
    )
    setups_dir: Path = Field(
        default_factory=lambda: PathConfig._default_setups_dir(),
        title="Setups Directory",
        description="Directory for setup-related files.",
    )
    annotations_template_path: Path = Field(
        default_factory=lambda: PathConfig._default_annotations_template_path(),
        title="Annotations Template Path",
        description="Path for annotations templates.",
    )

    @staticmethod
    def _default_outputs_dir() -> Path:
        return PathConfig._get_repo_dir() / "outputs"

    @staticmethod
    def _default_io_dir() -> Path:
        return PathConfig._get_repo_dir() / "io"

    @staticmethod
    def _default_datasets_dir() -> Path:
        return PathConfig._default_io_dir() / "datasets"

    @staticmethod
    def _default_experiments_root_dir() -> Path:
        return Path.home() / "io" / "experiments"

    @staticmethod
    def _default_models_dir() -> Path:
        return Path.home() / "io" / "models"

    @staticmethod
    def _default_visualizations_dir() -> Path:
        return PathConfig._default_outputs_dir() / "visualization"

    @staticmethod
    def _default_logs_dir() -> Path:
        return PathConfig._default_outputs_dir() / "logs"

    @staticmethod
    def _default_setups_dir() -> Path:
        return PathConfig._default_outputs_dir() / "setups"

    @staticmethod
    def _default_annotations_template_path() -> Path:
        return PathConfig._get_src_dir() / "annotations"

    @staticmethod
    def _get_src_dir() -> Path:
        """
        Determines the source directory based on the current file location.
        """
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def _get_repo_dir() -> Path:
        """
        Determines the repository root directory based on the source directory.
        """
        return PathConfig._get_src_dir().parent

    def add_src_to_sys_path(self) -> None:
        """
        Adds the source directory to the Python system path for imports.
        """
        src_dir = str(self._get_src_dir())
        if src_dir not in sys.path:
            sys.path.append(src_dir)


__all__ = ["PathConfig"]
