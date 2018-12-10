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
from keras.datasets import mnist
from keras.regularizers import l1, l2
from functools import partial

def PermanentDropout(p_fail):
  """ Make dropout work when using predict(), not only on train """
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

# preparing dataset
def get_mnist(out_scaler = 1.0, in_scaler = 255.):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 ** 2) / 255. * in_scaler
    x_test = x_test.reshape(-1, 28 ** 2) / 255. * in_scaler
    digits = {x: [out_scaler if y == x else 0. for y in range(10)] for x in range(10)}
    y_train = np.array([digits[y] for y in y_train])
    y_test = np.array([digits[y] for y in y_test])
    return x_train, y_train, x_test, y_test

def create_random_weight_model(Ns, p_fails, p_bound, KLips, func = 'sigmoid', reg_type = 0, reg_coeff = 0.01, train_dropout_l1 = 0):
  """ Create some simple network with given dropout prob, weights and Lipschitz coefficient for sigmoid """
  
  # creating model
  model = Sequential()

  # clearing the variable
  errors = {0: 0} 

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
    elif reg_type == 0 or reg_type == 'dropout':
        regularizer = lambda w : 0

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
  last = len(Ns) - 2
  return model, errors[last] if last in errors else None, errors

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
