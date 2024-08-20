import torch


EPSILON = 1e-8

class PairwiseDistanceCalculator:
    def __init__(self, metric: str):
        self.metric = metric
        self.metric_function = self._get_metric_function()

    def _get_metric_function(self):
        metric_function_map = {
            'l2': self._l2_distance,
            'cosine': self._cosine_distance,
            'dot': self._dot_product,
        }
        if self.metric not in metric_function_map:
            raise ValueError(f'Unsupported similarity function: {self.metric}')
        return metric_function_map[self.metric]

    def _l2_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            x.unsqueeze(1).expand(-1, y.shape[0], -1) -
            y.unsqueeze(0).expand(x.shape[0], -1, -1)
        ).pow(2).sum(dim=2)

    def _cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        cosine_similarities = (normalised_x.unsqueeze(1) * normalised_y.unsqueeze(0)).sum(dim=2)
        return 1 - cosine_similarities

    def _dot_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -(x.unsqueeze(1) * y.unsqueeze(0)).sum(dim=2)

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.metric_function(x, y)
