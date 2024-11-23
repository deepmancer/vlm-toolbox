from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Type, Union, List
from enum import Enum


class Validator:
    """A utility class for validating various types of inputs."""

    @staticmethod
    def validate_list_item(item: Any, valid_list: Iterable[Any], name: Optional[str] = None) -> None:
        """Validate that an item is present in a given list."""
        if item not in valid_list:
            Validator._raise_value_error(item, valid_list, name)

    @staticmethod
    def validate_iterable_item(item: Any, iterable: Iterable[Any], name: Optional[str] = None) -> None:
        """Validate that an item is present in a given iterable."""
        if item not in iterable:
            Validator._raise_value_error(item, iterable, name)

    @staticmethod
    def validate_dict_key(key: Any, dictionary: Mapping[Any, Any], name: Optional[str] = None) -> None:
        """Validate that a key exists in a given dictionary."""
        if key not in dictionary:
            Validator._raise_value_error(key, list(dictionary.keys()), name)

    @staticmethod
    def validate_enum_value(value: Any, enum_class: Type[Enum], name: Optional[str] = None) -> None:
        """Validate that a value is a member of the specified Enum."""
        if not isinstance(value, enum_class):
            Validator._raise_value_error(value, [e.value for e in enum_class], name)

    @staticmethod
    def validate_list_of_enum_values(values: Iterable[Any], enum_class: Type[Enum], name: Optional[str] = None) -> None:
        """Validate that all values in a list are members of the specified Enum."""
        for value in values:
            Validator.validate_enum_value(value, enum_class, name)

    @staticmethod
    def validate_non_negative_int(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is a non-negative integer."""
        if not isinstance(value, int) or value < 0:
            Validator._raise_value_error(value, "a non-negative integer", name)

    @staticmethod
    def validate_positive_int(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            Validator._raise_value_error(value, "a positive integer", name)

    @staticmethod
    def validate_infinity(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is positive infinity."""
        if value != float("inf"):
            Validator._raise_value_error(value, "infinity", name)

    @staticmethod
    def validate_positive_number(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is a positive number (int or float)."""
        if not isinstance(value, (int, float)) or value <= 0:
            Validator._raise_value_error(value, "a positive number", name)

    @staticmethod
    def validate_bool(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is a boolean."""
        if not isinstance(value, bool):
            Validator._raise_value_error(value, "either True or False", name)

    @staticmethod
    def validate_optional_string(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is either None or a string."""
        if value is not None and not isinstance(value, str):
            Validator._raise_value_error(value, "either None or a string", name)

    @staticmethod
    def validate_dict(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is a dictionary."""
        if not isinstance(value, Mapping):
            Validator._raise_value_error(value, "a dictionary", name)

    @staticmethod
    def validate_iterable(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is an iterable."""
        if not isinstance(value, Iterable):
            Validator._raise_value_error(value, "an iterable object", name)

    @staticmethod
    def validate_path_exists(path: Union[str, Path], name: Optional[str] = None) -> None:
        """Validate that a filesystem path exists."""
        if not Path(path).exists():
            name_part = f" '{name}'" if name else ""
            raise ValueError(f"Invalid path{name_part}: '{path}' does not exist.")

    @staticmethod
    def validate_string(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is a string."""
        if not isinstance(value, str):
            Validator._raise_value_error(value, "a string", name)

    @staticmethod
    def validate_string_value(value: Any, valid_values: Iterable[Any], name: Optional[str] = None) -> None:
        """Validate that a string value is within a set of valid values."""
        if value not in valid_values:
            Validator._raise_value_error(value, valid_values, name)

    @staticmethod
    def validate_not_falsy(value: Any, name: Optional[str] = None) -> None:
        """Validate that a value is not falsy."""
        if not value:
            Validator._raise_value_error(value, "a truthy value", name)

    @staticmethod
    def validate_all(validators: Iterable, *args: Any) -> None:
        """Validate using all provided validator functions."""
        for validator in validators:
            validator(*args)

    @staticmethod
    def validate_any(validators: Iterable, *args: Any) -> None:
        """Validate using any one of the provided validator functions."""
        errors: List[str] = []
        for validator in validators:
            try:
                validator(*args)
                return
            except ValueError as e:
                errors.append(str(e))
        error_messages = "; ".join(errors)
        raise ValueError(f"None of the validators passed for inputs {args}: {error_messages}")

    @staticmethod
    def _raise_value_error(value: Any, expected: Any, name: Optional[str]) -> None:
        """Raise a ValueError with a standardized message."""
        name_part = f" '{name}'" if name else ""
        raise ValueError(f"Invalid input{name_part}: {value}. Expected: {expected}.")

__all__ = ["Validator"]
