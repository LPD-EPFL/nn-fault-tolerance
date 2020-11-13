### Make matrix "continuous" in the first dimension
# developed in FilterPlayground.ipynb

from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import tensorflow as tf
import findiff

def gkern(size = 20, nsig = 3, normalize = 'max'):
    """Returns a 2D Gaussian kernel."""

    assert normalize in ['max', '?'], "Normalize: max -- make max value to 1, ? -- for size-dependent to normlize the integral"

    if size == 2 and normalize == 'max':
        return [-1, 1]

    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    x = np.linspace(-nsig, nsig, size+1)
    kern1d = np.diff(st.norm.cdf(x))

    # normalizing so that max == 1
    #kern1d = kern1d / np.max(kern1d)

    # normalizing over length
    kern1d = kern1d / kern1d.sum()

    # difference kernel -> integral is 0
    #kern1d = kern1d - kern1d.sum() / size

    #kern1d[size // 2] += 1

    return kern1d

def convolved(W, kernel):
    """ Input: W n_out x n_in, kernel: 1D array
        Output: operation of convolution of W along each n_in
        output shape: like W
    """

    # W as batch (n_in -- for each neuron x 1 channel x n_out -- to convolve)
    v = tf.reshape(tf.transpose(W), (W.shape[1], 1, W.shape[0]))

    # filter array: W, channels
    filt = tf.reshape(tf.constant(kernel, dtype = tf.float32), (-1, 1, 1))

    # returning reshaped convolution n_in x n_out
    z = tf.nn.convolution(v, filt,padding = 'SAME', data_format = 'NCW')
    z = tf.reshape(z, (W.shape[1], W.shape[0]))
    z = tf.transpose(z)
    return z

def conv_loss(W, kernel, reduce = tf.reduce_sum, subtract = False, do_print = False):
    """ Loss for a kernel (smoothness) """
    # invalid shape -> no loss
    if W.shape[0] < len(kernel) or len(kernel) <= 1: return tf.constant(0.0)
#    print("CONV", W.shape, len(kernel))
    if do_print:
        plt.plot(kernel)
        print('iCi', np.sum(np.multiply(kernel, np.arange(len(kernel)))))
        print('Ci', np.sum(kernel))
    z = convolved(W, kernel)
    if subtract:
        z = z - W
    z = tf.abs(z)
    z = reduce(z, axis = 0)
    z = tf.reduce_sum(z)
    return z

def derkern(size = 5):
    """ Difference kernel (derivative approximation )"""
    if size == 0:
        return [-1, 1]
    size = size * 2
    return findiff.coefficients(deriv = 1, acc = size)['center']['coefficients']

def aggregate(dct):
    """ Aggregate dict of losses into one average loss """
    return sum(dct.values()) / len(dct)

def smoothness_scale_free(W):
    """ Input: matrix W: n_out x n_in
         Assume that W ~ 1/n_in
         Will return a tensor computing smoothness of W in n_out
    """

    # sizes
    n_out, n_in = map(lambda x : x.value, W.shape)

    # kernel sizes for derivative part
    sizes = [0, 2, 5, 7]

    # smoothness coefficients for convolution part
    conv_sizes = [n_out // 20, n_out // 10]

    # derivatives: calculate the derivative in the array
    regs_derivative = {size: conv_loss(W, derkern(size = size)) for size in sizes}

    # convolution to obtain the mean - original -> non-smoothness
    regs_smoothness = {size: conv_loss(W, gkern(size = size), subtract = True, reduce = tf.reduce_mean)
                   for size in conv_sizes}

    return {'derivative': aggregate(regs_derivative), 'smoothness': aggregate(regs_smoothness)}
