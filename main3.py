'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Deconvolution2D, Reshape
from keras.models import Model
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

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


"""
# MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train.shape

image_Size = X_train.shape[1]
X_train.shape[1]
image_Size
original_Dim = image_Size * image_Size
X_train = np.reshape(X_train, [-1, original_Dim])
"""
PATH = os.getcwd()

train_path = PATH+'/fruits/fruits-360/train'
train_batch = os.listdir(train_path)
x_train = []
x_test = []
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
x_train, x_test, _, __ = train_test_split(DATA, dada, test_size=0.20)

# finally converting list into numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)

x_train.shape
image_size = x_train.shape[1]
image_size
original_dim = image_size * image_size
#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train.shape
x_test.shape

# network parameters
input_shape = (original_dim, )
input_shape
intermediate_dim = 512
batch_size = 1
latent_dim = 16
epochs = 20

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(128,128,1), name='encoder_input')
inputs.shape
x = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128,128,1))(inputs)
x.shape
x = MaxPooling2D((2, 2), padding='same', input_shape=(128,128,32))(x)
x.shape
x = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64,64,32))(x)
x.shape
x = MaxPooling2D((2, 2), padding='same', input_shape=(64,64,64))(x)
x.shape
x = Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32,32,64))(x)
x.shape
x = MaxPooling2D((2, 2), padding='same', input_shape=(32,32,128))(x)
x.shape

x = Dense(intermediate_dim, activation='relu', input_shape=(16,16,32))(x)
x.shape
y = Dense(intermediate_dim, activation='relu', input_shape=(intermediate_dim,))(x)
y.shape
y = Flatten(input_shape=(intermediate_dim,))(y)
z_mean = Dense(latent_dim, name='z_mean', input_shape=(intermediate_dim,))(y)
z_mean.shape
z_log_var = Dense(latent_dim, name='z_log_var', input_shape=(intermediate_dim,))(y)
z_log_var.shape
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
z.shape

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
latent_inputs.shape
x = Dense(intermediate_dim, activation='relu', input_shape=(latent_dim,))(latent_inputs)
x.shape
yy = Dense(intermediate_dim, activation='relu', input_shape=(intermediate_dim,))(x)
yy.shape
yy = Reshape((16,16,2))(yy)
#y.reshape(8,8,8)
yy.shape
x = Deconvolution2D(16, (3, 3), activation='relu', padding='same', input_shape=(16,16,2))(yy)
x.shape
x = UpSampling2D((2, 2), input_shape=(16,16,16))(x)
x.shape
x = Deconvolution2D(8, (3, 3), activation='relu', padding='same', input_shape=(32,32,16))(x)
x.shape
x = UpSampling2D((2, 2), input_shape=(32,32,8))(x)
x.shape
x = Deconvolution2D(3, (3, 3), activation='relu', padding='same', input_shape=(64,64,8))(x)
x.shape
x = UpSampling2D((2, 2), input_shape=(64,64,3))(x)
x.shape
outputs = Deconvolution2D(1, (3, 3), activation='sigmoid', padding='same', input_shape=(128,128,3))(x)
outputs.shape
outputs = UpSampling2D((1, 1), input_shape=(128,128,1))(outputs)
outputs.shape
#outputs = Dense(original_dim, activation='sigmoid')(y)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
#decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    #y_test = x_test
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    reconstruction_loss
    kl_loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    #vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    if args.weights:
        vae.load_weights('fruits_weights.h5')
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save('new_fruits.h5')
        encoder.save('new_fruits_encoder.h5')
        decoder.save('new_fruits_decoder.h5')
        vae.save_weights('new_fruits_weights.h5')




encoded_imgs = encoder.predict(x_test)
decoded_imgs = vae.predict(x_test)
decoded_imgS = decoder.predict(encoded_imgs[2])
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i+20].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i+20].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.imshow(x_test[39].reshape(128, 128))
plt.imshow(decoded_imgs[39].reshape(128, 128))
plt.imshow(decoded_imgS[21].reshape(128, 128))








#############################################
#############################################

img = cv2.imread('FramesPos/2014_05_01_22_25_17_545000_.tif')
img = cv2.imread('data/test/2014_05_01_22_07_32_277000_.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (470, 470))
img = img.flatten()
img = np.array([img])
img = img/255

coded = encoder.predict(img)
predicted = decoder.predict(coded[2])


plt.imshow(img[0].reshape(470, 470))
plt.imshow(predicted[0].reshape(470, 470))
x_test[50].shape
img[0].shape
coded.size
