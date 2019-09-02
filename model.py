# for environ
import os

# only using device 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# importing tensorflow
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def init_tf_keras():
  """ Initialize TensorFlow """
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  set_session(sess)
  print("Initialized TensorFlow")

# to use only the memory that we need
init_tf_keras()

# standard imports
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Lambda
from keras.initializers import Constant
from keras.regularizers import l1, l2
from numbers import Number
from helpers import *
from keras import Model, Input

class Continuous(keras.regularizers.Regularizer):
    """ Regularizer penalizing for weights being too different for neurons. Specifically, 
        computes 1/n_l^2 \sum \sum_{ij} |W_{ij}-W_{i+1,j}| """

    def __init__(self, beta = 1e-2):
        """ Initialize (just save parameters) """
        self.beta = beta

    def __call__(self, x):
        """ Regularize x """
        # transposed weights n_{l-1} x n_l
        W_T = x

        # non-transposed weight matrix n_l x n_{l-1}
        W = tf.transpose(W_T)

        # out shape
        n_l = W.shape[0].value

        # resulting regularizer
        reg = tf.reduce_sum(tf.abs(W[1:,] - W[:-1,:])) / n_l ** 2

        # result = beta * reg
        return self.beta * reg
        
    def get_config(self):
        """ Get parameters """
        return {'beta': self.beta}

class Balanced(keras.regularizers.Regularizer):
    """Regularizer for (wmax/wmin)^2 for W_i = \sum\limits_j |W_ij|, see main paper, Eq. 1
    # Arguments
        mu: Float; the multiplier (regularization parameter)
        eps: Float, the value to add in the denomenator so it does not blow up (should be smaller than the typical weight)
    """

    def __init__(self, mu = 0.0, eps = 1e-5):
        # just saving the parameters as floats
        self.mu = K.cast_to_floatx(mu)
        self.eps = K.cast_to_floatx(eps)

    def __call__(self, x):
        """ Compute regularization for input matrix x """
        # taking the absolute value
        x = K.abs(x)

        # summing over the first axis (Keras uses transposed w.r.t. our notation matrices)
        x = K.sum(x, axis = 0)

        # squaring to comply with Eq.1
        x = K.square(x)

        # resulting regularization
        regularization = self.mu * K.max(x) / (self.eps + K.min(x))
        return regularization

    def get_config(self):
        return {'mu': float(self.mu), 'eps': float(self.eps)}

def IdentityLayer(input_shape=None):
     """ A layer which does nothing """
     return Lambda(
         lambda x: x + 0, input_shape=input_shape, name='Identity')

def IndependentCrashes(p_fail, input_shape = None):
  """ Make dropout work when using predict(), not only on train, without scaling """
  assert isinstance(p_fail, Number), "pfail must be a number"
  return Lambda(lambda x: K.dropout(x, level=p_fail) * (1 - p_fail), input_shape = input_shape, name = 'Crashes')

def get_custom_activation(KLips, func):
  """ Get custom sigmoid activation with given Lipschitz constant """
  assert isinstance(KLips, Number), "KLips must be a number"
  def custom_activation(x):
    if func == 'sigmoid':
        return K.sigmoid(4 * KLips * x)
    elif func == 'relu':
        return K.relu(KLips * x)
    else: raise NotImplementedError("Activation function %s is not supported" % str(func))
  return custom_activation

def create_fc_crashing_model(Ns, weights, biases, p_fail, KLips = 1, func = 'sigmoid', reg_type = None, reg_coeff = 0, do_print = True, loss = keras.losses.mean_squared_error, optimizer = None, do_compile = True):
  """ Create a simple network with given dropout prob, weights and Lipschitz coefficient for sigmoid
      Ns: array of shapes: [input, hidden1, hidden2, ..., output]
      weights: array with matrices. The shape must be [hidden1 x input, hidden2 x hidden1, ..., output x hiddenLast]
      biases: array with vectors. The shape must be [hidden1, hidden2, ..., output]
      p_fail: array with p_fail for [input, hidden1, ..., output]. Must be the same size as Ns. Both for inference and training
      KLips: the Lipschitz coefficient
      func: The acivation function. Currently 'relu' and 'sigmoid' are supported. Note that the last layer is linear to agree with the When Neurons Fail article
      reg_type: The regularization type 'l1' or 'l2'
      reg_coeff: The regularization parameter
  """

  # default optimizer
  if not optimizer:
    optimizer = keras.optimizers.Adadelta()
  
  # input sanity check
  assert isinstance(Ns, list), "Ns must be a list"
  assert_equal(len(Ns), len(p_fail), "Shape array length", "p_fail array length")
  assert_equal(len(Ns), len(weights) + 1, "Shape array length", "Weights array length + 1")
  assert_equal(len(biases), len(weights), "Biases array length", "Weights array length")
  assert func in ['relu', 'sigmoid'], "Activation %s must be either relu or sigmoid" % str(func)
  assert reg_type in [None, 'l1', 'l2', 'balanced', 'continuous'], "Regularization %s must be either l1, l2 or None" % str(reg_type)
  assert isinstance(KLips, Number), "KLips %s must be a number" % str(KLips)
  assert isinstance(reg_coeff, Number), "reg_coeff %s must be a number" % str(reg_coeff)

  # creating model
  model = Sequential()

  # loop over shapes
  for i in range(len(Ns)):
    # is the first layer (with input)?
    is_input = (i == 0)

    # is the last layer (with output)?
    is_output = (i == len(Ns) - 1)

    # probability of failure for this shape
    p = p_fail[i]

    # current shape
    N_current = Ns[i]

    # previous shape or None
    N_prev = Ns[i - 1] if i > 0 else None

    # adding a dense layer if have previous shape (otherwise it's input)
    if not is_input:
      # deciding the type of regularizer
      if reg_type == 'l2':
          regularizer = l2(reg_coeff)
      elif reg_type == 'l1':
          regularizer = l1(reg_coeff)
      elif reg_type == 'balanced':
          # only doing it for first layer (where the crashes are!)
          if i == 1:
              regularizer = Balanced(reg_coeff)
          else:
              regularizer = lambda w : 0
      elif reg_type == 'continuous':
          # only doing it for first layer (where the crashes are!)
          if i == 1:
              regularizer = Continuous(reg_coeff)
          else:
              regularizer = lambda w : 0
      elif reg_type == None:
          regularizer = lambda w : 0
      else:
          raise(NotImplementedError("Regularization type"))

      # deciding the activation function
      activation = 'linear' if is_output else get_custom_activation(KLips, func)

      # extracting weights and biases
      w = weights[i - 1]
      b = biases[i - 1]
      assert_equal(w.shape, (N_current, N_prev), "Weight matrix %d/%d shape" % (i, len(Ns) - 1), "Ns array entries")
      assert_equal(b.shape, (N_current, ), "Biases vector %d/%d shape" % (i, len(Ns) - 1), "Ns array entry")

      # adding a Dense layer
      model.add(Dense(N_current, input_shape = (N_prev, ), kernel_initializer = Constant(w.T),
          activation = activation, bias_initializer = Constant(b), kernel_regularizer = regularizer))

    # adding dropout if needed
    if p > 0:
      model.add(IndependentCrashes(p, input_shape = (N_current, )))

  # parameters for compilation
  parameters = {'loss': loss, 'optimizer': optimizer, 'metrics': [keras.metrics.categorical_accuracy, 'mean_squared_error', 'mean_absolute_error']}

  # if compilation requested, doing it
  if do_compile:
    # compiling the model
    model.compile(**parameters)

    # printing the summary
    if do_print: model.summary()

    # returning Keras model
    return model

  # otherwise returning the parameters for compilation
  else:
    return model, parameters

def faulty_model(model, p_inference):
    """ Add crashes to every layer of the model """
    
    assert len(model.inputs) == 1, "Model must have exactly one input"
    
    # obtaining input/output shape 
    in_shape = model.inputs[0].shape[1:]

    # creating duplicate input
    inp = Input(shape = in_shape)
    
    def faulty_net(inp, model = None, p_inference = None):
        """ Obtain faulty output for input tensor inp """
        
        # current input
        z = inp

        # sanity check for input
        assert type(p_inference) == list, "p_inference must be a list"
        assert len(model.layers) == len(p_inference), "p must be present for every layer, now have |p|=%d and L=%d" % (
            len(p_inference), len(model.layers))

        # applying layers one-by-one
        for p, layer in zip(p_inference, model.layers):
            # applying layer
            z = layer.call(z)

            # adding independent crashes
            z = IndependentCrashes(p, input_shape = z.shape)(z)

        # outputting the result
        return z

    # model with crashes at each layer
    return Model(inputs = inp, outputs = Lambda(partial(faulty_net, model = model, p_inference = p_inference))(inp))
