import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Lambda
from keras.initializers import Constant
from functools import partial

def PermanentDropout(p_fail):
  """ Make dropout work when using predict(), not only on train """
  return Lambda(lambda x: K.dropout(x, level=p_fail))

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

def get_custom_activation(KLips):
  """ Get custom sigmoid activation with given Lipschitz constant """
  def custom_activation(x):
    return K.sigmoid(4 * KLips * x)
  return custom_activation

def create_random_weight_model(Ns, KLips):
  """ Create some simple network with given dropout prob, weights and Lipschitz coefficient for sigmoid """
  
  # creating model
  model = Sequential()

  # adding layers
  for i in range(len(Ns) - 1):
    
    # adding dense layer with sigmoid for hidden and linear for last layer
    model.add(Dense(Ns[i + 1], input_shape = (Ns[i], ),
                    kernel_initializer = 'random_normal',
                    activation = get_custom_activation(KLips),
                    bias_initializer = 'random_normal'))

  model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

  #model.summary()
  return model

def create_model(p_fails, layer_weights, layer_biases, KLips):
  """ Create some simple network with given dropout prob, weights and Lipschitz coefficient for sigmoid """
  
  # checking if length matches
  assert(len(p_fails) == len(layer_weights))
  assert(len(layer_biases) == len(layer_weights))
  
  # creating model
  model = Sequential()
  
  # adding layers
  for i, (p_fail, w, b) in enumerate(zip(p_fails, layer_weights, layer_biases)):
    # is last layer (with output)?
    is_last = i + 1 == len(p_fails)
    
    # adding dense layer with sigmoid for hidden and linear for last layer
    model.add(Dense(w.shape[1], input_shape = (w.shape[0], ),
                    kernel_initializer = Constant(w),
                    activation = get_custom_activation(KLips),
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
