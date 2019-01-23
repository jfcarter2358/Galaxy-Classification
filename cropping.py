# import statements
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
def get_delta(arr):
    delta = np.zeros((len(arr), len(arr[0])))
    for i in range(1, len(arr) - 1):
        for j in range(1, len(arr[0]) - 1):
            x_1 = arr[i][j]
            x_2 = arr[i][j]
            y_1 = arr[i][j]
            y_2 = arr[i][j]
            # left x value
            if i > 1:
                x_1 = arr[i - 1][j]
            # right x value
            if i < len(arr):
                x_2 = arr[i + 1][j]
            # top y value
            if j > 1:
                y_1 = arr[i][j - 1]
            # bottom y value
            if j < len(arr[0]):
                y_2 = arr[i][j + 1]
            del_x = x_1 - x_2
            del_y = y_1 - y_2
            delta[i][j] = np.sqrt(del_x ** 2 + del_y ** 2)
    return delta

@jit(nopython=True)
def get_contig(arr, cell_size, relative, offset):
    contig = np.zeros((int(len(arr) / cell_size), int(len(arr[0]) / cell_size)))
    mid_x = int(len(arr) / 2)
    mid_y = int(len(arr[0]) / 2)
    coord_x = int(mid_x / 4)
    coord_y = int(mid_y / 4)
    
    max_l = 0
    for i in range(-offset * 4 + mid_x, offset * 4 + mid_x):
        for j in range(-offset * 4 + mid_y, offset * 4 + mid_y):
            if arr[i][j] > max_l:
                max_l = arr[i][j]
    
    threshold = relative * max_l
    
    for i in range(-offset, offset):
        for j in range(-offset, offset):            
            contig[int(mid_x / cell_size) + i][int(mid_y / cell_size) + j] = 1

    for coord_x in range(int(mid_x / 4), 0, -1):
        for coord_y in range(int(mid_y / 4), 0, -1):
            c = True
            for i in range(0, cell_size):
                for j in range(0, cell_size):
                    x = coord_x * 4 + i
                    y = coord_y * 4 + j
                    if arr[x][y] < threshold:
                        c = False
            if c == True:
                for i in range(-2, 2):
                    for j in range(-2, 2):
                        if (coord_x + i) > -1 and (coord_y + j) > -1 and (coord_x + i) < len(contig) - 1 and (coord_y + j) < len(contig[0]):
                            if contig[coord_x + i][coord_y + j] > 0:
                                contig[coord_x][coord_y] = 1
    
    for coord_x in range(int(mid_x / 4), 0, -1):
        for coord_y in range(int(mid_y / 4), int(len(arr[0]) / 4), 1):
            c = True
            for i in range(0, cell_size):
                for j in range(0, cell_size):
                    x = coord_x * 4 + i
                    y = coord_y * 4 + j
                    if arr[x][y] < threshold:
                        c = False
            if c == True:
                for i in range(-2, 2):
                    for j in range(-2, 2):
                        if (coord_x + i) > -1 and (coord_y + j) > -1 and (coord_x + i) < len(contig) - 1 and (coord_y + j) < len(contig[0]):
                            if contig[coord_x + i][coord_y + j] > 0:
                                contig[coord_x][coord_y] = 1
                
    for coord_x in range(int(mid_x / 4), int(len(arr) / 4), 1):
        for coord_y in range(int(mid_y / 4), 0, -1):
            c = True
            for i in range(0, cell_size):
                for j in range(0, cell_size):
                    x = coord_x * 4 + i
                    y = coord_y * 4 + j
                    if arr[x][y] < threshold:
                        c = False
            if c == True:
                for i in range(-2, 2):
                    for j in range(-2, 2):
                        if (coord_x + i) > -1 and (coord_y + j) > -1 and (coord_x + i) < len(contig) - 1 and (coord_y + j) < len(contig[0]):
                            if contig[coord_x + i][coord_y + j] > 0:
                                contig[coord_x][coord_y] = 1
        
    for coord_x in range(int(mid_x / 4), int(len(arr) / 4), 1):
        for coord_y in range(int(mid_y / 4), int(len(arr[0]) / 4), 1):
            c = True
            for i in range(0, cell_size):
                for j in range(0, cell_size):
                    x = coord_x * 4 + i
                    y = coord_y * 4 + j
                    if arr[x][y] < threshold:
                        c = False
            if c == True:
                for i in range(-2, 2):
                    for j in range(-2, 2):
                        if (coord_x + i) > -1 and (coord_y + j) > -1 and (coord_x + i) < len(contig) - 1 and (coord_y + j) < len(contig[0]):
                            if contig[coord_x + i][coord_y + j] > 0:
                                contig[coord_x][coord_y] = 1
    return contig

@jit(nopython=True)
def get_lum(im, lx, ly):
    im_d = np.zeros((lx, ly))
    for x in range(0, lx):
        for y in range(0, ly):
            im_d[x][y] = np.sqrt(im[x][y][0] ** 2 + im[x][y][1] ** 2 + im[x][y][2] ** 2)
    return im_d

url = 'data/images_training_rev1/images_training_rev1'
files_all = [f for f in os.listdir(url)]
step = 100

for a in range(0, len(files_all), step):
    try:
        files = files_all[a:a + step]
    except:
        files = files_all[a:]

    ims = []
    titles = []
    s_time = datetime.now()

    print('---- {} : {} ----'.format(a / step, int(len(files_all) / step)))
    
    l = len(files)

    # print('---- step 0 :: image loading ----')
    
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        # load the image and and append it to the list of images
        im = plt.imread('data/images_training_rev1/images_training_rev1/' + files[i], format='jpeg')
        ims.append(im)
        # add the name of the file for tracking purposes
        titles.append(files[i])

    l_x = len(im)
    l_y = len(im[0])
    mid_x = int(len(im) / 2)
    mid_y = int(len(im[0]) / 2)
    xs = [i for i in range(0, len(im))]
    ys = [i for i in range(0, len(im[0]))]
    X, Y = np.meshgrid(xs, ys)

    # print('---- step 1 :: raw processing ----')
    
    ims_d = []
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        ims_d.append(get_lum(ims[i], l_x, l_y))

    sigma_y = 8.0
    sigma_x = 8.0
    sigma = [sigma_y, sigma_x]
    sims_d = []

    # print('---- step 2 :: smoothing ----')

    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        sims_d.append(sp.ndimage.filters.gaussian_filter(ims_d[i], sigma, mode='constant'))
    
    # print('---- step 3 :: gradient processing ----')
    
    grads_d = []
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        grads_d.append(get_delta(sims_d[i]))
    
    # print('---- step 4 :: contiguous processing ----')
    
    contigs = []
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        contigs.append(get_contig(grads_d[i], cell_size=4, relative=0.5, offset=2))
    crops_x = []
    crops_y = []

    # print('---- step 5 :: cropping preprocessing ----')
    
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        min_x = len(contigs[i])
        max_x = 0
        min_y = len(contigs[i][0])
        max_y = 0
        for x in range(0, len(contigs[i])):
            for y in range(0, len(contigs[i][0])):
                if contigs[i][x][y] == 1:
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
        crops_x.append([min_x * 4, max_x * 4])
        crops_y.append([min_y * 4, max_y * 4])

    cropped = []

    # print('---- step 6 :: cropping ----')
    
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        min_x = int(crops_x[i][0])
        max_x = int(crops_x[i][1])
        min_y = int(crops_y[i][0])
        max_y = int(crops_y[i][1])
        i_t = []
        for ii in ims[i][min_x:max_x]:
            i_t.append(ii[min_y:max_y])
        cropped.append(i_t)

    cs_d = []

    # print('---- step 7 :: galaxy cleaning preprocessing ----')
    
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        cs_d.append(get_lum(cropped[i], len(cropped[i]), len(cropped[i][0])))

    # print('---- step 8 :: galaxy cleaning ----')
    
    pure_contigs = []
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        pure_contigs.append(get_contig(cs_d[i], cell_size=4, relative=0.2, offset=2))

     #print('---- step 9 :: purifying ----')

    pure_images = []
    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        pc = pure_contigs[i]
        c = cs_d[i]
        for x in range(0, len(c)):
            for y in range(0, len(c[0])):
                if pc[int(x / 4)][int(y / 4)] == 0:
                    c[x][y] = 0
        pure_images.append(c)
        
    finished = []

    # print('---- step A :: resizing images ----')

    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        p = pure_images[i]
        blank = np.zeros((l_x, l_y))
        for x in range(0, len(p)):
            for y in range(0, len(p[0])):
                min_x = int(crops_x[i][0])
                min_y = int(crops_y[i][0])
                blank[x + min_x][y + min_y] = p[x][y]
        finished.append(blank)

    # print('---- step B :: saving images ----')

    for i in range(0, l):
        # if (i % 10) == 0:
        #     print('{}...{}'.format(l, i))
        img_s = Image.fromarray(finished[i]).convert('RGB')
        img_s.save('./cropped/{}'.format(titles[i]))

    e_time = datetime.now()
    print((e_time - s_time).seconds)