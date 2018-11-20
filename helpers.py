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

# todo: support for updateable C
def get_kernel_reg(layer, errors, is_last, C = 1., p = 0.1, KLips = 1., lambda_ = 0.1):
    """ Get a Erf regularizer for layer"""
    
    def kernel_reg(w, layer = layer, is_last = is_last, C_layer = C, p_layer = p, KLips = KLips, lambda_ = lambda_):
        """ Regularizer for a layer """
        # Maximal 1-norm over output neuron
        if layer == 0: return 0
        wnorm1 = K.max(K.sum(K.abs(w), axis = 0))
        
        # error (induction)
        error = (p_layer * C_layer + KLips * (1 - p_layer) * errors[layer - 1]) * wnorm1
        
        # saving the error for the next call
        errors[layer] = error

#        print("Error is_last = %d %d = (pC + K(1-p) DeltaOld)wnorm p = %f C = %s K = %f DeltaOld = %s wnorm = %s" % (is_last, layer, p_layer, str(C_layer), KLips, str(errors[layer - 1]), str(wnorm1)))
        
        # returning the error scaled
        return error * lambda_ if is_last else 0
    
    # returning the function
    return kernel_reg

def get_kernel_reg_v2(layer, errors, is_last, C, p, KLips = 1., lambda_ = 0.1):
    """ Get a DeltaNetwork regularizer for layer"""

    def kernel_reg(w, layer = layer, is_last = is_last, C_layer = C, p_layer = p, KLips = KLips, lambda_ = lambda_):
        """ Regularizer for a layer """
        # Maximal 1-norm over output neuron
        if layer == 0: return 0
        W = K.abs(w)

        #print("Error is_last = %d %d = W(pC + K(1-p) DeltaOld) p = %f C = %s K = %f DeltaOld = %s W = %s" % (is_last, layer, p_layer, str(C_layer), KLips, str(errors[layer - 1]), str(W)))
        
        # error (induction)
        error = K.dot(K.transpose(W), p_layer * C_layer + KLips * (1 - p_layer) * errors[layer - 1])
        
        # saving the error for the next call
        errors[layer] = error

        # returning the error scaled
        return K.mean(error) * lambda_ if is_last else 0

    # returning the function
    return kernel_reg

def create_random_weight_model(Ns, p_fails, p_bound, KLips, func = 'sigmoid', reg_type = 0, reg_coeff = 0.01, C_arr = [], C_per_neuron_arr = [], train_dropout_l1 = 0):
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
    C_arr += [K.variable(1.0 if func == 'sigmoid' else 0.0)]
    C_per_neuron_arr += [K.variable(np.array([1. if func == 'sigmoid' else 0.] * Ns[i + 1]).reshape(-1, 1))]
    N_pre = Ns[i]
    N_post = Ns[i + 1]

    print(N_pre, N_post)

    if reg_type == 'l2':
        regularizer = l2(reg_coeff)
    elif reg_type == 'l1':
        regularizer = l1(reg_coeff)
    elif reg_type  == 'delta':
        regularizer = get_kernel_reg(i, errors, is_last, KLips = KLips, lambda_ = reg_coeff, C = C_arr[i - 1], p = p_bound[i])
    elif reg_type  == 'delta_network':
        regularizer = get_kernel_reg_v2(i, errors, is_last, KLips = KLips, lambda_ = reg_coeff, C = C_per_neuron_arr[i - 1], p = p_bound[i])
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
