class BaseConfig:
    config = {}

    @staticmethod
    def get_config(*args, **kwargs):
        raise NotImplementedError

    
    @classmethod
    def match_key(cls, key, strict=False):
        if key in cls.config:
            return key

        if strict:
            return None
          
        for k in cls.config:
            if k in key:
                return k

        return None

    
    @classmethod
    def is_valid(cls, key, strict=False):
        if cls.match_key(key, strict=strict):
            return True
        return False

    @classmethod
    def get(cls, key, strict=False):
        matched_key = cls.match_key(key, strict=strict)
        return cls.config.get(matched_key, None)
