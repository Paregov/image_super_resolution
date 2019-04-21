import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.models import Model
from keras.losses import mean_squared_error
import tensorflow as tf

# Define the VGG19 model for all of the loss functions

# vgg19.get_output_at('block2_pool')
# vgg19.get_layer('block2_pool')

BATCH_SIZE = 16
CHANNELS = 3
SHAPE_LR = 32
NF = 64
VGG_MEAN = np.array([123.68, 116.779, 103.939])  # RGB
GAN_FACTOR_PARAMETER = 2.


def normalize(v):
    # v.get_shape().assert_has_rank(4)
    return v / K.mean(v, axis=[1, 2, 3], keepdims=True)

# From:
# https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
def gram_matrix_old(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# From
# https://becominghuman.ai/utilising-cnns-to-transform-your-model-into-a-budding-artist-1330dc392e25
def gram_matrix_(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



def gram_matrix(v):
    v.get_shape().assert_has_rank(4)
    dim = v.get_shape().as_list()
    v = K.reshape(v, [-1, dim[1] * dim[2], dim[3]])
    return K.matmul(v, v, transpose_a=True)


# TODO: Create another function
# It gave me better results with
def perceptual_loss(y_true, y_pred):
    vgg = VGG19(input_shape=(128,128,3), include_top=False)

    pool2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_pool').output)
    pool5 = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_pool').output)
    pool2.trainable = False
    pool5.trainable = False

    pool2_loss = K.mean(K.square(pool2(y_true) - pool2(y_pred)))
    pool5_loss = K.mean(K.square(pool5(y_true) - pool5(y_pred)))

    return (0.2 * pool2_loss) + (0.02 * pool5_loss)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def get_phis(vgg, x):
    vgg(x)

    pool2 = vgg.layers[6]
    pool5 = vgg.layers[21]
    print('pool2: ')
    print(pool2)
    print('pool5: ')
    print(pool5)

    return pool2, pool5


# 6 - pool2
# 21 - pool5
def perceptual_loss_pat(y_true, y_pred):
    vgg19 = VGG19(include_top=False)
    vgg19.trainable = False
    #  = normalize(pool2)
    # pool5 = normalize(pool5)

    phi_a_1, phi_a_2 = get_phis(vgg19, y_true)
    phi_b_1, phi_b_2 = tf.split(vgg19, y_pred)

    pool2_loss = mean_squared_error(phi_a_1, phi_b_1)
    pool5_loss = mean_squared_error(phi_a_2, phi_b_2)

    return pool2_loss + pool5_loss


def texture_loss(x, p=16):
    _, h, w, c = x.get_shape().as_list()
    x = normalize(x)
    assert h % p == 0 and w % p == 0
    # logger.info('Create texture loss for layer {} with shape {}'.format(x.name, x.get_shape()))

    x = K.space_to_batch_nd(x, [p, p], [[0, 0], [0, 0]])  # [b * ?, h/p, w/p, c]
    x = K.reshape(x, [p, p, -1, h // p, w // p, c])       # [p, p, b, h/p, w/p, c]
    x = K.transpose(x, [2, 3, 4, 0, 1, 5])                # [b * ?, p, p, c]
    patches_a, patches_b = tf.split(x, 2, axis=0)          # each is b,h/p,w/p,p,p,c

    patches_a = K.reshape(patches_a, [-1, p, p, c])       # [b * ?, p, p, c]
    patches_b = K.reshape(patches_b, [-1, p, p, c])       # [b * ?, p, p, c]
    return mean_squared_error(
        gram_matrix(patches_a),
        gram_matrix(patches_b))


def perceptual_loss_16(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_pool').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

