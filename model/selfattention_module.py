import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Lambda,BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Add, Multiply, Concatenate, Input, MaxPooling2D,Conv1D
from tensorflow.keras import backend as K

def SE(x_in, r=8):
    """ Squeeze and Excitation """
    channels = x_in.shape[-1]
    # Squeeze
    x = GlobalAveragePooling2D()(x_in)
    # Excitation
    x = Dense(channels//r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([x_in, x])


def SimAM(x, v_lambda=1e-4):
    """ SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks 
        https://github.com/ZjjConan/SimAM
        https://github.com/cpuimage/SimAM/blob/main/SimAM.py <- tensorflow
    """
    # spatial size 
    n = x.shape[1] * x.shape[2] - 1 
    # square of (t - u) 
    d = tf.math.square(x - tf.reduce_mean(x,axis=(1, 2), keepdims=True))
    # d.sum() / n is channel variance 
    v = tf.reduce_sum(d,axis=(1,2), keepdims=True) / n 
    # E_inv groups all importance of X 
    E_inv = d / (4 * (v + v_lambda)) + 0.5 
    return x * tf.sigmoid(E_inv)


def SRM(x):
    """ SRM : A Style-based Recalibration Module for Convolutional Neural Networks 
        https://github.com/taki0112/SRM-Tensorflow
    """
    _, h, w, c = x.get_shape().as_list()
    bs = tf.shape(x)[0]

    channels = c

    x = tf.reshape(x, shape=[bs, -1, c]) # [bs, h*w, c]

    x_mean, x_var = tf.nn.moments(x, axes=1, keepdims=True) # [bs, 1, c]
    x_std = tf.sqrt(x_var + 1e-5)

    t = tf.concat([x_mean, x_std], axis=1) # [bs, 2, c]

    z = Conv1D(channels, kernel_size=2, strides=1)(t)
    z = BatchNormalization(momentum=0.9, epsilon=1e-05, center=True, scale=True)(z)

    g = tf.sigmoid(z)
    x = tf.reshape(x * g, shape=[bs, h, w, c])
    
    return x

