class BaseMetric:
    name = None

    def compute(self, true_labels, predicted_labels, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.name

    def __str__(self):
        return f"Metric(name={self.name})"
    
    def __repr__(self):
        return self.__str__()
