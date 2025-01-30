from transformers import DefaultDataCollator


class BaseDataCollator(DefaultDataCollator):
    def get_label_names(self):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
