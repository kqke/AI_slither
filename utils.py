import numpy as np


def l1_distance(p1, p2):
    y1, x1 = p1
    y2, x2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def sample_bool_matrix(mat):
    indices = np.argwhere(mat)
    n = indices.shape[0]
    i = np.random.randint(n)
    loc = tuple(indices[i].tolist())
    return loc
