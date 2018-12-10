import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

init_tf_keras()

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Lambda
from keras.initializers import Constant
from keras.regularizers import l1, l2
from functools import partial
from numbers import Number

def IndependentCrashes(p_fail):
  """ Make dropout work when using predict(), not only on train, without scaling """
  return Lambda(lambda x: K.dropout(x, level=p_fail) * (1 - p_fail))

# calculate first norm
norm1 = partial(np.linalg.norm, ord = 1)

# calculate second norm
norm2 = partial(np.linalg.norm, ord = 2)

def dot_abs(x, y):
  """ Dot product between absolute values of vectors x, y """
  return np.dot(np.abs(x), np.abs(y))

def norm1_minus_dot_abs(x, y):
  """ Product of first norms - dot product between absolute values """
  return norm1(x) * norm1(y) - dot_abs(x, y)

def get_custom_activation(KLips, func):
  """ Get custom sigmoid activation with given Lipschitz constant """
  def custom_activation(x):
    if func == 'sigmoid':
        return K.sigmoid(4 * KLips * x)
    elif func == 'relu':
        return K.relu(KLips * x)
  return custom_activation

def assert_equal(x, y, name_x = "x", name_y = "y"):
  """ Assert that x == y and if not, pretty-print the error """
  assert x == y, "%s = %s must be equal to %s = %s" % (str(name_x), str(x), str(name_y), str(y))

def create_fixed_weight_model(Ns, weights, biases, p_fail_inference, p_fail_train, KLips = 1, func = 'sigmoid', reg_type = None, reg_coeff = 0):
  """ Create a simple network with given dropout prob, weights and Lipschitz coefficient for sigmoid
      Ns: array of shapes: [input, hidden1, hidden2, ..., output]
      weights: array with matrices. The shape must be [hidden1 x input, hidden2 x hidden1, ..., output x hiddenLast]
      biases: array with vectors. The shape must be [hidden1, hidden2, ..., output]
      p_fail_inference: array with p_fail for [input, hidden1, ..., output]. Must be the same size as Ns
      p_fail_train: same for training phase
      KLips: the Lipschitz coefficient
      func: The acivation function. Currently 'relu' and 'sigmoid' are supported. Note that the last layer is linear to agree with the When Neurons Fail article
      reg_type: The regularization type 'l1' or 'l2'
      reg_coeff: The regularization parameter
  """
  
  # input sanity check
  assert_equal(len(Ns), len(p_fail_inference), "Shape array length", "p_fail inference array length")
  assert_equal(len(p_fail_train), len(p_fail_inference), "p_fail train array length", "p_fail inference array length")
  assert_equal(len(Ns), len(weights) + 1, "Shape array length", "Weights array length + 1")
  assert_equal(len(biases), len(weights), "Biases array length", "Weights array length")
  assert func in ['relu', 'sigmoid'], "Activation %s must be either relu or sigmoid" % func
  assert reg_type in [None, 'l1', 'l2'], "Regularization %s must be either l1, l2 or None" % reg_type
  assert 

  # creating model
  model = Sequential()

  # adding layers
  for i in range(len(Ns) - 1):
    # is last layer (with output)?
    is_last = i + 2 == len(Ns)
    p_fail = p_fails[1 + i]
    N_pre = Ns[i]
    N_post = Ns[i + 1]

    print(N_pre, N_post)

    if reg_type == 'l2':
        regularizer = l2(reg_coeff)
    elif reg_type == 'l1':
        regularizer = l1(reg_coeff)
    elif reg_type == None:
        regularizer = lambda w : 0
    else:
        raise(NotImplementedError("Regularization type"))

    # adding dense layer with sigmoid for hidden and linear for last layer
    model.add(Dense(Ns[i + 1], input_shape = (Ns[i], ),
                    kernel_initializer = Constant(np.random.randn(N_pre, N_post) * np.sqrt(2. / N_pre) / KLips),
                    activation = 'linear' if is_last else get_custom_activation(KLips, func),
                    bias_initializer = 'random_normal',
                    kernel_regularizer = regularizer
                   ))

    # adding dropout to all layers but last
    if not is_last and p_fail > 0:
      model.add(PermanentDropout(p_fail))

    if i == 0 and train_dropout_l1 > 0:
      model.add(PermanentDropout(train_dropout_l1))

  model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy', 'mean_squared_error'])

  model.summary()
  return model

def create_model(p_fails, layer_weights, layer_biases, KLips, func = 'sigmoid'):
  """ Create some simple network with given dropout prob, weights and Lipschitz coefficient for sigmoid """
  
  # checking if length matches
  assert(len(p_fails) == len(layer_weights))
  assert(len(layer_biases) == len(layer_weights))
  
  # creating model
  model = Sequential()

  # adding layers
  for i, (p_fail, w, b) in enumerate(zip(p_fails[1:] + [0], layer_weights, layer_biases)):
    # is last layer (with output)?
    is_last = i + 1 == len(layer_weights)

    # adding dense layer with sigmoid for hidden and linear for last layer
    model.add(Dense(w.shape[1], input_shape = (w.shape[0], ),
                    kernel_initializer = Constant(w),
                    activation = 'linear' if is_last else get_custom_activation(KLips, func),
                    bias_initializer = Constant(b)))
    
    # adding dropout to all layers but last
    if not is_last and p_fail > 0:
      model.add(PermanentDropout(p_fail))
  
  # compiling model with some loss and some optimizer (they are unused)
  model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
  #model.summary()
  return model

def generate_params(**kwargs):
    """ Arguments -> array of dicts """
    
    # fetched the last parameter
    if len(kwargs) == 0:
        yield {}
        return
    
    # some argument
    param = list(kwargs.keys())[0]
    
    # the rest of the dictionary
    kwargs1 = {x: y for x, y in kwargs.items() if x != param}
    
    # loop over kwargs data
    for val in kwargs[param]:
        # loop over experiments
        for res in generate_params(**kwargs1):
            res[param] = val
            yield {x: y for x, y in res.items()}

def rank_loss(a, b):
    """ For given a, b compute the average number of misordered pairs, O(n^2) """
    
    # flattening data
    a, b = np.array(a).flatten(), np.array(b).flatten()
    
    # checking shape
    assert len(a) == len(b), "Lengths must agree"
    
    # sorting b in order of a
    b = np.array(b)[np.argsort(a)]
    
    # number of bad pairs
    res = sum([sum([1 if i < j and x >= y else 0 for j, y in enumerate(b)]) for i, x in enumerate(b)])
    
    # total number of pairs
    NN = len(a) * (len(a) - 1) / 2
    
    # return the ratio
    return 1. * res / NN

def compute_rank_losses(data, key):
    """ Compute rank losses for a dict with data, referenced to key """
    return {keyother: rank_loss(data[key], data[keyother]) for keyother in data.keys() if keyother != key}
