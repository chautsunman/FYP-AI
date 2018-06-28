import numpy as np

def get_linear_data(n):
    w = np.array([2, 16])
    b = 18
    x = np.random.randn(n, 2)
    y = np.dot(x, w) + b
    return x, y
