from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.models import Model
from keras.losses import mean_squared_error
import tensorflow as tf


def normalize(v):
    assert len(v.shape) == 4
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
    assert len(v.shape) == 4
    dim = v.shape
    v = K.reshape(v, [-1, dim[1] * dim[2], dim[3]])
    return tf.matmul(v, v, transpose_a=True)


def texture_loss_calculation(y_true, y_pred, p=16):
    _, h, w, c = y_pred.shape
    assert h % p == 0 and w % p == 0

    y_true = normalize(y_true)
    y_pred = normalize(y_pred)

    # logger.info('Create texture loss for layer {} with shape {}'.format(x.name, x.get_shape()))

    y_true = tf.space_to_batch_nd(y_true, [p, p], [[0, 0], [0, 0]])           # [b * ?, h/p, w/p, c]
    y_pred = tf.space_to_batch_nd(y_pred, [p, p], [[0, 0], [0, 0]]) # [b * ?, h/p, w/p, c]

    y_true = K.reshape(y_true, [p, p, -1, h // p, w // p, c])                 # [p, p, b, h/p, w/p, c]
    y_pred = K.reshape(y_pred, [p, p, -1, h // p, w // p, c])       # [p, p, b, h/p, w/p, c]

    patches_truth = tf.transpose(y_true, [2, 3, 4, 0, 1, 5])                  # [b * ?, p, p, c]
    patches_prediction = tf.transpose(y_pred, [2, 3, 4, 0, 1, 5])             # [b * ?, p, p, c]

    patches_a = K.reshape(patches_truth, [-1, p, p, c])                      # [b * ?, p, p, c]
    patches_b = K.reshape(patches_prediction, [-1, p, p, c])                      # [b * ?, p, p, c]

    mse = mean_squared_error(gram_matrix(patches_a), gram_matrix(patches_b))

    return tf.reduce_mean(mse)


def perceptual_loss(y_true, y_pred):
    vgg = VGG19(input_shape=(128, 128, 3), include_top=False)

    pool2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_pool').output)
    pool5 = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_pool').output)
    pool2.trainable = False
    pool5.trainable = False

    pool2_loss = K.mean(K.square(normalize(pool2(y_true)) - normalize(pool2(y_pred))))
    pool5_loss = K.mean(K.square(normalize(pool5(y_true)) - normalize(pool5(y_pred))))

    pool2_loss = tf.reduce_mean(pool2_loss)
    pool5_loss = tf.reduce_mean(pool5_loss)

    return (0.2 * pool2_loss) + (0.02 * pool5_loss)


def perceptual_loss_16(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_pool').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


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
    # pool2 = normalize(pool2)
    # pool5 = normalize(pool5)

    phi_a_1, phi_a_2 = get_phis(vgg19, y_true)
    phi_b_1, phi_b_2 = tf.split(vgg19, y_pred)

    pool2_loss = mean_squared_error(phi_a_1, phi_b_1)
    pool5_loss = mean_squared_error(phi_a_2, phi_b_2)

    return pool2_loss + pool5_loss


def texture_loss(y_true, y_pred):
    return texture_loss_calculation(y_true, y_pred)


def texture_loss_multi_layers(y_true, y_pred):
    vgg = VGG19(input_shape=(128, 128, 3), include_top=False)

    conv1_1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv1').output)
    conv2_1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv1').output)
    conv3_1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv1').output)

    conv1_1.trainable = False
    conv2_1.trainable = False
    conv3_1.trainable = False

    conv1_1_loss = texture_loss_calculation(conv1_1(y_true), conv1_1(y_pred))
    conv2_1_loss = texture_loss_calculation(conv2_1(y_true), conv2_1(y_pred))
    conv3_1_loss = texture_loss_calculation(conv3_1(y_true), conv3_1(y_pred))

    return (1 * conv1_1_loss) + (1 * conv2_1_loss) + (1 * conv3_1_loss)


def perceptual_plus_texture_loss(y_true, y_pred):
    perceptual = perceptual_loss(y_true, y_pred)
    texture = texture_loss_multi_layers(y_true, y_pred)

    return perceptual + texture

