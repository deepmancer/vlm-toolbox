from pydantic import BaseModel, ConfigDict
import yaml
from typing import Any, Dict


class BaseConfig(BaseModel):
    """
    Base configuration class for the vision-language toolbox.

    This class provides a standardized structure for configurations and supports
    YAML file-based instantiation.

    Attributes:
        model_config (ConfigDict): Settings for model configuration:
            - from_attributes (bool): Populate the dictionary from attributes.
            - validate_assignment (bool): Validate assignments to the dictionary.
            - populate_by_name (bool): Populate the dictionary by name.
            - extra (str): Defines handling of extra fields. Default is "allow".
            - arbitrary_types_allowed (bool): Allow arbitrary types in the configuration.
    """
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_yaml(cls, yaml_file_path: str) -> "BaseConfig":
        """
        Create an instance of the configuration class from a YAML file.

        Args:
            yaml_file_path (str): Path to the YAML file.

        Returns:
            BaseConfig: An instance of the BaseConfig class populated with data
                        from the YAML file.

        Raises:
            FileNotFoundError: If the specified YAML file does not exist.
            yaml.YAMLError: If there is an error in parsing the YAML file.
        """
        try:
            with open(yaml_file_path, 'r', encoding='utf-8') as file:
                yaml_data = yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file '{yaml_file_path}' was not found.") from e
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing the YAML file: {yaml_file_path}.") from e

        # Filter yaml_data to only include keys defined in the model class
        valid_data = {
            key: value for key, value in yaml_data.items() if key in cls.__annotations__
        }

        return cls(**valid_data)

__all__ = ["BaseConfig"]
