from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Deconvolution2D, Reshape
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.preprocessing import image
from keras import backend as K
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import math


encoder = load_model('fruits_encoder.h5')
decoder = load_model('fruits_decoder.h5')

PATH = os.getcwd()

train_path = PATH+'/fruits/fruits-360/train'
train_batch = os.listdir(train_path)
dada = [0 for i in range(6006)]
DATA=[]
# if data are in form of images
for sample in train_batch:
    img_path = train_path+'/'+sample
    #print(img_path)
    x = cv2.imread(img_path)
    x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (128, 128))
    x = np.expand_dims(x, axis=2)
    #x = x.flatten()
    #x = image.load_img(img_path)
	# preprocessing if required
    DATA.append(x)

DATA = np.array(DATA)
DATA = DATA.astype('float32') / 255

encoded_imgs = encoder.predict(DATA)
decoded_imgS = decoder.predict(encoded_imgs[2])

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(DATA[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgS[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

plt.imshow(DATA[19].reshape(128, 128))
plt.imshow(decoded_imgS[19].reshape(128, 128))
