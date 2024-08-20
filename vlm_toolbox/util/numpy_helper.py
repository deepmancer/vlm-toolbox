import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def low_entropy_softmax(x):
    scale = np.sqrt(x.shape[0])
    adjusted_x = x / scale
    return softmax(adjusted_x)
