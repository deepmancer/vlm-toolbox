def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


def get_nested_attr(obj, attr_path, default=None):
    attrs = attr_path.split('.')
    for attr in attrs:
        obj = getattr(obj, attr, default)
        if obj is default:
            return default
    return obj

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
