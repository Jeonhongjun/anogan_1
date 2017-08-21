from __future__ import print_function
from collections import defaultdict
# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.layers import Input, Reshape, Flatten
from keras.layers import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.generic_utils import Progbar
import numpy as np

K.set_image_data_format('channels_last')


class Generator(object):
    """docstring for Generator."""
    def __init__(self, latent_shape=(100, ), n_filters=64):
        super(Generator, self).__init__()
        # save initial args
        self.latent_shape = latent_shape
        self.n_filters = n_filters
        # generator input of latent space vector Z, typically a 1D vector
        gen_input = Input(shape=self.latent_shape)
        # layer 1 - higher dimensional dense layer
        cnn = Dense(1024)(gen_input)
        cnn = Activation('relu')(cnn)
        # layer 2 - higer dimensional dense layer
        cnn = Dense(7 * 7 * 128)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        # transform 1D to 3D matrix (2D image plus channels)
        cnn = Reshape((7, 7, 128))(cnn)
        # layer 3 - convulational layer - filter matching
        cnn = UpSampling2D(size=(2, 2))(cnn)
        cnn = Conv2D(self.n_filters, 5, padding='same')(cnn)
        cnn = Activation('relu')(cnn)
        # layer 4 - convulational layer - channel reducer
        cnn = UpSampling2D(size=(2, 2))(cnn)
        cnn = Conv2D(1, 5, padding='same')(cnn)
        gen_output = Activation('tanh')(cnn)

        self.model = Model(gen_input, gen_output)


class Discriminator(object):
    """docstring for Discriminator."""
    def __init__(self, input_shape=(28, 28, 1), n_filters=64):
        super(Discriminator, self).__init__()
        # save initial args
        self.input_shape = input_shape
        self.n_filters = n_filters

        disc_input = Input(shape=self.input_shape)

        cnn = Conv2D(self.n_filters, 5, padding='same')(disc_input)
        cnn = LeakyReLU()(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

        cnn = Conv2D(self.n_filters * 2, 5, padding='same')(cnn)
        cnn = LeakyReLU()(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

        cnn = Flatten()(cnn)

        cnn = Dense(1024)(cnn)
        cnn = LeakyReLU()(cnn)

        cnn = Dense(1)(cnn)
        disc_output = Activation('sigmoid')(cnn)

        self.model = Model(disc_input, disc_output)


class AnoGan(object):
    """docstring for AnoGan."""
    def __init__(self, input_shape=(28, 28, 1), latent_shape=(100, ), n_filters=64):
        super(AnoGan, self).__init__()

        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.n_filters = n_filters
        self.models = {}

        self.generator = Generator(latent_shape=self.latent_shape, n_filters=self.n_filters)
        self.discriminator = Discriminator(input_shape=self.input_shape, n_filters=self.n_filters)

        gan_input = Input(shape=self.latent_shape)
        h = self.generator.model(gan_input)
        self.discriminator.model.trainable = False
        gan_output = self.discriminator.model(h)

        self.model = Model(gan_input, gan_output)

        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        e2e_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

        self.generator.model.compile(loss='binary_crossentropy', optimizer="SGD")
        self.model.compile(loss='binary_crossentropy', optimizer=e2e_optim)

        self.discriminator.model.trainable = True
        self.discriminator.model.compile(loss='binary_crossentropy', optimizer=d_optim)

    def train(self, X_train, epochs, batch_size, X_test=None, verbose=0):
        self.epochs = epochs
        self.train_history = defaultdict(list)
        for epoch in range(self.epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))

            n_iter = int(X_train.shape[0]/batch_size)
            progress_bar = Progbar(target=n_iter)

            epoch_gen_loss = []
            epoch_disc_loss = []

            for idx in range(n_iter):
                progress_bar.update(idx, force=True)
                noise = np.random.uniform(-1, 1, size=(batch_size, self.latent_shape[0]))
                image_batch = X_train[idx * batch_size:(idx + 1) * batch_size]
                generated_images = self.generator.model.predict(noise, verbose=verbose)
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)
                d_loss = self.discriminator.model.train_on_batch(X, y)
                epoch_disc_loss.append(d_loss)
                noise = np.random.uniform(-1, 1, (2 * batch_size, self.latent_shape[0]))
                self.discriminator.model.trainable = False
                # we want to train the generator to trick the discriminator
                # For the generator, we want all the {fake, not-fake}
                # labels to say not-fake
                g_loss = self.model.train_on_batch(noise, np.ones(2 * batch_size))
                self.discriminator.model.trainable = True
                epoch_gen_loss.append(g_loss)

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

            # generate an epoch report on performance
            self.train_history['generator'].append(generator_train_loss)
            self.train_history['discriminator'].append(discriminator_train_loss)

            ROW_FMT = '{0:<22s} | {1:<15.3f}'
            print('\n{0:<22s} | {1:15s}'.format('component', 'loss'))
            print('-' * 30)
            print(ROW_FMT.format('generator (train)', self.train_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)', self.train_history['discriminator'][-1]))

            # generate some digits to display
            noise = np.random.uniform(-1, 1, size=(100, self.latent_shape[0]))
            # get a batch to display
            generated_images = self.generator.model.predict(noise, verbose=verbose)
            # arrange them into a grid
            img = (np.concatenate([r.reshape(-1, 28)
                                   for r in np.split(generated_images, 10)
                                   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(img).save('plot_epoch_{0:03d}_generated.png'.format(epoch + 1))
