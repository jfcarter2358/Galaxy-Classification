import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.ndimage
import os
from PIL import Image
from numba import jit
from datetime import datetime

@jit(nopython=True)
def red(a, b):
    for i in range(148, 276):
        for j in range(148, 276):
            b[i - 148][j - 148] = a[i][j][0]
    return b

url = 'classification_data/Processed/'
files = [f for f in os.listdir(url)]

for f in files:
    a = plt.imread(url + f, format='jpeg')
    b = np.zeros((128, 128))
    b = red(a, b)
    c = Image.fromarray(b).convert('RGB')
    c.save('./reduced/{}.jpg'.format(f))