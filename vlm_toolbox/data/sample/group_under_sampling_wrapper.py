import numpy as np
from sklearn.utils import _safe_indexing


class GroupUnderSamplingWrapper:
    def __init__(self, under_sampler):
        self.under_sampler = under_sampler
        
    def fit_resample(self, X, y, y_group=None):
        if y_group is None:
            return self.under_sampler.fit_resample(X, y)
        
        X_resampled, _ = self.under_sampler.fit_resample(X, np.array(y_group))
        return X_resampled, _safe_indexing(y, self.under_sampler.sample_indices_)
