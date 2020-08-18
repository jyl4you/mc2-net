import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)

class InstanceNormalization(keras.layers.Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = keras.layers.InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer':  keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer':  keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint':   keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint':  keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


""" Motion correction network for MC2-Net """


class Encoder(keras.Model):
    def __init__(self, initial_filters=64):
        super(Encoder, self).__init__()

        self.filters = initial_filters

        self.conv1 = keras.layers.Conv2D(self.filters, kernel_size=7, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(self.filters*2, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(self.filters*4, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.n1 = InstanceNormalization()
        self.n2 = InstanceNormalization()
        self.n3 = InstanceNormalization()


    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.n1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.n2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.n3(x, training=training)
        x = tf.nn.relu(x)

        return x


class Residual(keras.Model):
    def __init__(self, initial_filters=256):
        super(Residual, self).__init__()

        self.filters = initial_filters

        self.conv1 = keras.layers.Conv2D(self.filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(self.filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.in1 = InstanceNormalization()
        self.in2 = InstanceNormalization()

    def call(self, x, training=True):
        inputs = x

        x = self.conv1(x)
        # x = self.in1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        # x = self.in2(x, training=training)
        x = tf.nn.relu(x)

        x = tf.add(x, inputs)

        return x


class Decoder(keras.Model):
    def __init__(self, initial_filters=128):
        super(Decoder, self).__init__()

        self.filters = initial_filters

        self.conv1 = keras.layers.Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding='same',
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2DTranspose(self.filters//2, kernel_size=3, strides=2, padding='same',
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(1, kernel_size=7, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.in1 = InstanceNormalization()
        self.in2 = InstanceNormalization()
        self.in3 = InstanceNormalization()

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.in1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.in2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.in3(x, training=training)
        x = tf.nn.relu(x)

        return x


class MC_Net(keras.Model):
    def __init__(self,
                 img_size=256,
                 num_filter=16,
                 num_contrast=3,
                 num_res_block=9):
        super(MC_Net, self).__init__()

        self.img_size = img_size
        self.filters = num_filter
        self.num_contrast = num_contrast
        self.num_res_block = num_res_block

        self.encoder_list = []
        for _ in range(num_contrast):
            self.encoder_list.append(Encoder(initial_filters=self.filters))

        self.res_block_list = []
        for _ in range(num_res_block):
            self.res_block_list.append(Residual(initial_filters=self.filters*4*num_contrast))

        self.decoder_list = []
        for _ in range(num_contrast):
            self.decoder_list.append(Decoder(initial_filters=self.filters*2))

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(MC_Net, self).build(input_shape)

    def call(self, x, training=True):
        x_list = []
        for i in range(self.num_contrast):
            x_list.append(self.encoder_list[i](x[i], training))
        x = tf.concat(x_list, axis=-1)

        for i in range(self.num_res_block):
            x = (self.res_block_list[i](x, training))

        y = tf.split(x, num_or_size_splits=self.num_contrast, axis=-1)

        y_list = []
        for i in range(self.num_contrast):
            y_list.append(self.decoder_list[i](y[i], training))

        return y_list


def ssim_loss(img1, img2):
    return -tf.math.log((tf.image.ssim(img1, img2, max_val=1.0)+1)/2)


def vgg_layers(layer_names):
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def vgg_loss(img1, img2, loss_model):
    img1 = tf.repeat(img1, 3, -1)
    img2 = tf.repeat(img2, 3, -1)

    return tf.reduce_mean(tf.square(loss_model(img1) - loss_model(img2)))


def make_custom_loss(l1, l2, loss_model):
    def custom_loss(y_true, y_pred):
        return l1*ssim_loss(y_true, y_pred) + l2*vgg_loss(y_true, y_pred, loss_model)

    return custom_loss