from keras.layers import LeakyReLU, Flatten, Dense
from keras.layers import Conv2D, UpSampling2D, add, Input
from keras.models import Model, Sequential
import tensorflow as tf
from layers import Resampling2D


model_weights_no_residual = './saved_models/weights.no.res.best.train.hdf5'
model_weights_with_residual = './saved_models/weights.with.res.best.train.hdf5'


def _residual_block(x, filters=64, kernel_size=3, kernel_initializer='uniform'):
    shortcut = x

    x = Conv2D(filters, kernel_size, padding='same', activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Conv2D(filters, kernel_size, padding='same', activation='linear', kernel_initializer=kernel_initializer)(x)

    return add([shortcut, x])


def _residual_network(x, kernel_initializer='uniform'):
    bicubic = Resampling2D([4, 4], method='BICUBIC')(x)
    
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer)(x)

    # Add the Residual layer 10 times
    for i in range(10):
        x = _residual_block(x, kernel_initializer=kernel_initializer)

    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer)(x)

    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer)(x)

    return add([x, bicubic])


def generator_no_residual(input_shape, summary=False, kernel_initializer='uniform'):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                     input_shape=input_shape, kernel_initializer=kernel_initializer))

    model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer))

    model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer))
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer))

    model.build()

    if summary:
        model.summary()

    return model


def generator_with_residual(input_shape, summary=False, kernel_initializer='uniform'):
    image_tensor = Input(shape=input_shape)
    network_output = _residual_network(image_tensor, kernel_initializer=kernel_initializer)
    model = Model(inputs=[image_tensor], outputs=[network_output])

    if summary:
        print(model.summary())

    return model


def discriminator(input_shape, summary=False, kernel_initializer='uniform'):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                             input_shape=input_shape, kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())
    model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())
    model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())
    model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())

    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())
    model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())

    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())
    model.add(Conv2D(filters=512, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='sigmoid'))

    if summary:
        model.summary()

    return model

