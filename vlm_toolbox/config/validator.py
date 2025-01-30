class Validator:
    @staticmethod
    def validate_enum_value(value, enum_class, name):
        if value not in enum_class.get_values():
            raise ValueError(f"Invalid {name}: {value}")

    @staticmethod
    def validate_list_of_enum_values(values, enum_class, name):
        for value in values:
            Validator.validate_enum_value(value, enum_class, name)

    @staticmethod
    def validate_non_negative_int(value, name):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError(f"Invalid {name}: {value}")

    @staticmethod
    def validate_positive_int(value, name):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"Invalid {name}: {value}")

    @staticmethod
    def validate_inf(value, name):
        if value != float("inf"):
            raise ValueError(f"Invalid {name}: {value}")
        
    @staticmethod
    def validate_positive_number(value, name):
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"Invalid {name}: {value}")

    @staticmethod
    def validate_bool(value, name):
        if not isinstance(value, bool):
            raise ValueError(f"Invalid {name}: {value}")

    @staticmethod
    def validate_optional_string(value, name):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Invalid {name}: {value}")

    @staticmethod
    def validate_dict(value, name):
        if not isinstance(value, dict):
            raise ValueError(f"Invalid {name}: {value}")
        
    @staticmethod
    def validate_all(validators, *args):
        for validator in validators:
            validator(*args)

    @staticmethod
    def validate_any(validators, *args):
        for validator in validators:
            try:
                validator(*args)
                return
            except ValueError:
                continue
        raise ValueError(f"None of the validators passed for inputs: {args}")
