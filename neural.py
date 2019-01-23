import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import sys
import collections
import pandas as pd

training_path = './classification_data/training'
testing_path = './classification_data/testing'

def get_label(p):
    l = p.split('.')[0]
    if l == '0':
        label = [1, 0, 0, 0]
    elif l == '1':
        label = [0, 1, 0, 0]
    elif l == '2':
        label = [0, 0, 1, 0]
    else:
        label = [0, 0, 0, 1]
    return label

def label_training_data():
    training_images = []
    for i in tqdm(os.listdir(training_path)):
        path = os.path.join(training_path, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        training_images.append([np.array(img), get_label(i)])
    shuffle(training_images)
    return training_images

def label_testing_data():
    testing_images = []
    testing_names = []
    for i in tqdm(os.listdir(testing_path)):
        testing_names.append(i)
        path = os.path.join(testing_path, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        testing_images.append([np.array(img), get_label(i)])
    return testing_images, testing_names

training_images = label_training_data()
testing_images, testing_names = label_testing_data()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

# sys.exit()

model = Sequential()

model.add(InputLayer(input_shape=[128, 128, 1]))

model.add(Conv2D(filters=32, kernel_size=5, strides=5, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=1, padding='same'))

model.add(Conv2D(filters=50, kernel_size=5, strides=5, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=1, padding='same'))

model.add(Conv2D(filters=80, kernel_size=5, strides=5, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=1, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(4, activation='softmax'))
# optimizer = Adam(lr=0.1)
optimizer = RMSprop(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=10, batch_size=100, shuffle=True)
model.summary()

with open('out.csv', 'w') as f:
    for i in range(0, len(tst_img_data)):
        data = tst_img_data[i].reshape(1, 128, 128, 1)
        model_out = model.predict([data])
        # print('{} :: {}'.format(model_out, tst_lbl_data[i]))
        for m in model_out[0]:
            f.write('{},'.format(m))
        for l in tst_lbl_data[i]:
            f.write('{},'.format(l))
        f.write(',{}'.format(testing_names[i]))
        f.write('\n')

df = pd.read_csv('out.csv', header=None)
df['pred'] = df[[0, 1, 2, 3]].idxmax(axis=1)
df['ans'] = df[[4, 5, 6, 7]].idxmax(axis=1)
df['ans'] -= 4
df['correct'] = (df['pred'] == df['ans'])
df.to_csv('check_neural.csv')