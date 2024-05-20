import time
import numpy as np
from scipy.signal import fftconvolve

a = np.ones((250,400))
b = np.zeros((880, 1040))
b[300:500, 400:500] = 255
t_s = time.time()
c1 = fftconvolve(b, a, mode='same')
print(time.time() - t_s)