import numpy as np


def l1_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def sample_bool_matrix(mat):
    ys, xs = np.where(mat)
    y = np.random.choice(ys)
    x = np.random.choice(xs)
    p = y, x
    return p
