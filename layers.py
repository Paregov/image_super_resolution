from enum import Enum
import tensorflow as tf

import keras.backend as K
from keras.utils import conv_utils
try:
    from keras.utils.conv_utils import normalize_data_format
except ImportError:
    from keras.backend.common import normalize_data_format
import keras.layers.convolutional as convolutional


class Resampling2D(convolutional.UpSampling2D):
    """Resampling layer for 2D inputs.
    Resizes the input images to ``size``.
    Parameters
    ----------
    size : int, or tuple of 2 integers.
        The size of the resampled image, in the format ``(rows, columns)``.
    method : Resampling2D.ResizeMethod or str, optional
        The resampling method to use. Default is
        ``Resampling2D.ResizeMethod.BILINEAR``. The string name of the enum
        may also be provided.
    data_format: str
        One of ``channels_last`` (default) or ``channels_first``. The ordering
        of the dimensions in the inputs. ``channels_last`` corresponds to
        inputs with shape ``(batch, height, width, channels)`` while
        ``channels_first`` corresponds to inputs with shape
        ``(batch, channels, height, width)``. It defaults to the
        ``image_data_format`` value found in your Keras config file at
        ``~/.keras/keras.json``. If you never set it, then it will be
        ``"channels_last"``.
    Examples
    --------
    >>> import numpy as np
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> import nethin.layers as layers
    >>> np.random.seed(42)
    >>>
    >>> X = np.array([[1, 2],
    ...               [2, 3]])
    >>> X = np.reshape(X, (1, 2, 2, 1))
    >>> resize = layers.Resampling2D((2, 2), data_format="channels_last")
    >>> inputs = Input(shape=(2, 2, 1))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> Y = model.predict_on_batch(X)
    >>> Y[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1. ,  1.5,  2. ,  2. ],
           [ 1.5,  2. ,  2.5,  2.5],
           [ 2. ,  2.5,  3. ,  3. ],
           [ 2. ,  2.5,  3. ,  3. ]], dtype=float32)
    >>> resize = layers.Resampling2D((0.5, 0.5), data_format="channels_last")
    >>> inputs = Input(shape=(4, 4, 1))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> X_ = model.predict_on_batch(Y)
    >>> X_[0, :, :, 0]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  2.],
           [ 2.,  3.]], dtype=float32)
    >>>
    >>> X = np.array([[1, 2],
    ...               [2, 3]])
    >>> X = np.reshape(X, (1, 1, 2, 2))
    >>> resize = layers.Resampling2D((2, 2), data_format="channels_first")
    >>> inputs = Input(shape=(1, 2, 2))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> Y = model.predict_on_batch(X)
    >>> Y[0, 0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1. ,  1.5,  2. ,  2. ],
           [ 1.5,  2. ,  2.5,  2.5],
           [ 2. ,  2.5,  3. ,  3. ],
           [ 2. ,  2.5,  3. ,  3. ]], dtype=float32)
    >>> resize = layers.Resampling2D((0.5, 0.5), data_format="channels_first")
    >>> inputs = Input(shape=(1, 4, 4))
    >>> outputs = resize(inputs)
    >>> model = Model(inputs, outputs)
    >>> X_ = model.predict_on_batch(Y)
    >>> X_[0, 0, :, :]  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  2.],
           [ 2.,  3.]], dtype=float32)
    """
    class ResizeMethod(Enum):
        BILINEAR = "BILINEAR"  # Bilinear interpolation
        NEAREST_NEIGHBOR = "NEAREST_NEIGHBOR"  # Nearest neighbor interpolation
        BICUBIC = "BICUBIC"  # Bicubic interpolation
        AREA = "AREA"  # Area interpolation

    def __init__(self,
                 size,
                 method=ResizeMethod.BILINEAR,
                 data_format=None,
                 **kwargs):

        super(Resampling2D, self).__init__(size=size,
                                           data_format=data_format,
                                           **kwargs)

        if isinstance(method, Resampling2D.ResizeMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = Resampling2D.ResizeMethod[method]
            except KeyError:
                raise ValueError("``method`` must be of type "
                                 "``Resampling2D.ResizeMethod`` or one of "
                                 "their string representations.")
        else:
            raise ValueError("``method`` must be of type "
                             "``Resampling2D.ResizeMethod`` or one of "
                             "their string representations.")

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_first":

            if input_shape[2] is not None:
                height = int(self.size[0] * input_shape[2] + 0.5)
            else:
                height = None

            if input_shape[3] is not None:
                width = int(self.size[1] * input_shape[3] + 0.5)
            else:
                width = None

            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)

        elif self.data_format == "channels_last":

            if input_shape[1] is not None:
                height = int(self.size[0] * input_shape[1] + 0.5)
            else:
                height = None

            if input_shape[2] is not None:
                width = int(self.size[1] * input_shape[2] + 0.5)
            else:
                width = None

            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):

        if self.method == Resampling2D.ResizeMethod.NEAREST_NEIGHBOR:
            return super(Resampling2D, self).call(inputs)

        else:
            if self.method == Resampling2D.ResizeMethod.BILINEAR:
                method = tf.image.ResizeMethod.BILINEAR
            elif self.method == Resampling2D.ResizeMethod.NEAREST_NEIGHBOR:
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            elif self.method == Resampling2D.ResizeMethod.BICUBIC:
                method = tf.image.ResizeMethod.BICUBIC
            elif self.method == Resampling2D.ResizeMethod.AREA:
                method = tf.image.ResizeMethod.AREA
            else:  # Should not be able to happen!
                raise ValueError("``method`` must be of type "
                                 "``Resampling2D.ResizeMethod`` or one of "
                                 "their string representations.")

            orig_shape = K.shape(inputs)
            if self.data_format == "channels_first":

                img_h = K.cast(orig_shape[2], K.floatx())
                img_w = K.cast(orig_shape[3], K.floatx())
                fac_h = K.constant(self.size[0], dtype=K.floatx())
                fac_w = K.constant(self.size[1], dtype=K.floatx())
                new_h = K.cast(img_h * fac_h + 0.5, "int32")
                new_w = K.cast(img_w * fac_w + 0.5, "int32")

                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                outputs = tf.image.resize_images(inputs,  # !TF
                                                 [new_h, new_w],
                                                 method=method,
                                                 align_corners=True)
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

                # new_shape = K.cast(orig_shape[2:], K.floatx())
                # new_shape *= K.constant(self.size, dtype=K.floatx())
                # new_shape = tf.to_int32(new_shape + 0.5)  # !TF
                # inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                # outputs = tf.image.resize_images(inputs, new_shape,  # !TF
                #                                  method=method,
                #                                  align_corners=True)
                # outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

            elif self.data_format == "channels_last":

                img_h = K.cast(orig_shape[1], K.floatx())
                img_w = K.cast(orig_shape[2], K.floatx())
                fac_h = K.constant(self.size[0], dtype=K.floatx())
                fac_w = K.constant(self.size[1], dtype=K.floatx())
                new_h = K.cast(img_h * fac_h + 0.5, "int32")
                new_w = K.cast(img_w * fac_w + 0.5, "int32")

                outputs = tf.image.resize_images(inputs,  # !TF
                                                 [new_h, new_w],
                                                 method=method,
                                                 align_corners=True)
            else:
                raise ValueError("Invalid data_format:", self.data_format)

        return outputs

    def get_config(self):

        config = {"method": self.method.name}

        base_config = super(Resampling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

