from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.models import Model
from keras.losses import mean_squared_error
import tensorflow as tf


def normalize(x):
    assert len(x.shape) == 4
    return x / K.mean(x, axis=[1, 2, 3], keepdims=True)


def gram_matrix(x):
    assert len(x.shape) == 4
    _, h, w, c = x.shape
    x = K.reshape(x, [-1, h * w, c])
    return tf.matmul(x, x, transpose_a=True)


def texture_loss_calculation(y_true, y_pred, p=16):
    _, h, w, c = y_pred.shape
    assert h % p == 0 and w % p == 0

    y_true = normalize(y_true)
    y_pred = normalize(y_pred)

    y_true = tf.space_to_batch_nd(y_true, [p, p], [[0, 0], [0, 0]])     # [b * ?, h/p, w/p, c]
    y_pred = tf.space_to_batch_nd(y_pred, [p, p], [[0, 0], [0, 0]])     # [b * ?, h/p, w/p, c]

    y_true = K.reshape(y_true, [p, p, -1, h // p, w // p, c])           # [p, p, b, h/p, w/p, c]
    y_pred = K.reshape(y_pred, [p, p, -1, h // p, w // p, c])           # [p, p, b, h/p, w/p, c]

    patches_truth = tf.transpose(y_true, [2, 3, 4, 0, 1, 5])            # [b * ?, p, p, c]
    patches_prediction = tf.transpose(y_pred, [2, 3, 4, 0, 1, 5])       # [b * ?, p, p, c]

    patches_a = K.reshape(patches_truth, [-1, p, p, c])                 # [b * ?, p, p, c]
    patches_b = K.reshape(patches_prediction, [-1, p, p, c])            # [b * ?, p, p, c]

    mse = mean_squared_error(gram_matrix(patches_a), gram_matrix(patches_b))

    return tf.reduce_mean(mse)


def perceptual_loss(y_true, y_pred):
    _, h, w, c = y_pred.shape
    vgg = VGG19(input_shape=(int(h), int(w), int(c)), include_top=False)

    pool2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_pool').output)
    pool5 = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_pool').output)
    pool2.trainable = False
    pool5.trainable = False
    pool2.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    pool5.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    pool2_loss = K.mean(K.square(normalize(pool2(y_true)) - normalize(pool2(y_pred))))
    pool5_loss = K.mean(K.square(normalize(pool5(y_true)) - normalize(pool5(y_pred))))

    pool2_loss = tf.reduce_mean(pool2_loss)
    pool5_loss = tf.reduce_mean(pool5_loss)

    return (2e-1 * pool2_loss) + (2e-2 * pool5_loss)


def perceptual_loss_16(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_pool').output)
    loss_model.trainable = False

    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def texture_loss(y_true, y_pred):
    return texture_loss_calculation(y_true, y_pred)


def texture_loss_multi_layers(y_true, y_pred):
    _, h, w, c = y_pred.shape
    vgg = VGG19(input_shape=(int(h), int(w), int(c)), include_top=False)

    conv1_1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv1').output)
    conv2_1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv1').output)
    conv3_1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv1').output)

    conv1_1.trainable = False
    conv2_1.trainable = False
    conv3_1.trainable = False

    conv1_1_loss = texture_loss_calculation(conv1_1(y_true), conv1_1(y_pred))
    conv2_1_loss = texture_loss_calculation(conv2_1(y_true), conv2_1(y_pred))
    conv3_1_loss = texture_loss_calculation(conv3_1(y_true), conv3_1(y_pred))

    return (3e-7 * conv1_1_loss) + (1e-6 * conv2_1_loss) + (1e-6 * conv3_1_loss)


def perceptual_plus_texture_loss(y_true, y_pred):
    perceptual = perceptual_loss(y_true, y_pred)
    texture = texture_loss_multi_layers(y_true, y_pred)

    return perceptual + texture

