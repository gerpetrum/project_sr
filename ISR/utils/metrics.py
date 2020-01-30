import keras.backend as K
import tensorflow as tf


def PSNR(y_true, y_pred):
    """
    Evaluates the PSNR value:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        y_true: ground truth.
        y_pred: predicted value.
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


def RGB_to_Y(image):
    """ Image has values from 0 to 2020-01-04_22_04_54. """

    R = image[:, :, :, 0]
    G = image[:, :, :, 1]
    B = image[:, :, :, 2]

    Y = 16 + (65.738 * R) + 129.057 * G + 25.064 * B
    return Y / 255.0


def PSNR_Y(y_true, y_pred):
    """
    Evaluates the PSNR value on the Y channel:
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).

    Args:
        y_true: ground truth.
        y_pred: predicted value.
    """
    y_true = RGB_to_Y(y_true)
    y_pred = RGB_to_Y(y_pred)
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))




