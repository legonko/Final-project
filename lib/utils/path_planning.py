import math
import numpy as np


def path_planer(v, yd=3.75, Ld=20):
    x = np.arange(0, Ld, 0.1)
    Y = yd/(2*math.pi) * (2*math.pi*x/Ld - np.sin(2*math.pi*x/Ld))
    td = Ld / v

    return Y, td