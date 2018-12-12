from helpers import *
from model import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.special import expit
import sys

# for obtaining current TF session
from keras.backend.tensorflow_backend import get_session

# for adding functions to Experiment class
__methods__ = []
register_method = register_method(__methods__)

# All functions assume only crashes at first layer

@register_method
def check_input_shape(self, data):
  """ Check that data is (nObj, nFeatures) """
  assert isinstance(data, np.ndarray), "Input must be an np.array"
  assert len(data.shape) == 2, "Input must be two-dimensional"
  assert data.shape[1] == self.N[0], "Input must be compliant with input shape (, %d)" % self.N[0]

@register_method
def check_p_layer0(self):
  """ Check that only have failures at first hidden layer output """
  assert all([p == 0 or i == 1 for i, p in enumerate(self.p_inference)]), "Must have failures only at first layer, other options are not implemented yet"

@register_method
def run_on_input(self, tensors, data):
  """ Run dict of tensors on input data """
  self.check_input_shape(data)

  # list of all keys, fixed order
  keys = list(tensors.keys())

  # running for all keys
  results = get_session().run([tensors[key] for key in keys], feed_dict = {self.model_correct.layers[0].input.name: data})

  # returning the result
  return {key: val for key, val in zip(keys, results)}

@register_method
def get_bound_v4(self, data):
  """ Exact error mean and std up to O(p^2) in case even if x_i are not small """

  self.check_p_layer0()

  @cache_graph(self)
  def get_graph():
    # layers of a correct network
    layers = self.model_correct.layers

    # need to drop all components one by one in the second layer input
    first_hidden_size = int(layers[1].input.shape[1])

    # results for each neuron on first hidden layer
    outputs = []

    # loop over first hidden layer neurons
    for i in range(first_hidden_size):
      # crashing i'th neuron only
      mask = [0 if i == j else 1 for j in range(first_hidden_size)]
  
      # data with one crash
      y = tf.multiply(layers[0].output, mask)
  
      # implementing the rest of the network
      for layer in layers[1:]:
        y = layer.activation(tf.matmul(y, layer.weights[0]) + layer.weights[1])

      # adding y_crashed - y_correct
      outputs.append(y - layers[-1].output)

    # std = sqrt(p * sum(outputs^2))
    # mean = -p * sum(outputs)
    p = self.p_inference[1]
    return {'mean': -p * sum(outputs), 'std': tf.sqrt(p * tf.reduce_sum(tf.square(outputs), axis = 0))}

  return self.run_on_input(get_graph(), data)
  
@register_method
def get_bound_v3(self, data):
  """ Exact error up to O(p^2x_i^2), assumes infinite width and small p """

  self.check_p_layer0()

  @cache_graph(self)
  def get_graph():
    # resulting gradient w.r.t. first layer output
    grad = []
    grad_sq = []

    # list of layers
    layers = self.model_correct.layers

    # for all output dimensions
    for output_dim in range(self.N[-1]):
      # get derivative of output
      out = layers[-1].output[:, output_dim]

      # w.r.t. first layer output
      grad    += [tf.reduce_sum(          tf.multiply(tf.gradients([out], [layers[0].output])[0], layers[0].output), axis = 1)]
      grad_sq += [tf.reduce_sum(tf.square(tf.multiply(tf.gradients([out], [layers[0].output])[0], layers[0].output)), axis = 1)]

    # compute the result
    p = self.p_inference[1]
    return {'mean': tf.transpose(tf.multiply(-p, grad)), 'std': tf.transpose(tf.sqrt(tf.multiply(p, grad_sq)))}

  return self.run_on_input(get_graph(), data)

@register_method
def get_bound_v2(self, data):
    """ Absolute values of matrices, mean/std """
    self.check_p_layer0()

    @cache_graph(self)
    def get_graph():
      # get input of the second layer network
      inp = tf.transpose(self.model_correct.layers[1].input)

      # get prob of failure
      p = self.p_inference[1]

      # get the product of all matrices (except first)
      R = tf.eye(self.N[-1], dtype = np.float32)
      Rsq = tf.eye(self.N[-1], dtype = np.float32)
      for w in self.W[1:][1::-1]:
        R = R @ np.abs(w)
        Rsq = Rsq @ np.square(w)

      # mean = p Rx, std^2 = p Rsq x^2
      return {'mean': p * tf.transpose(tf.matmul(R, inp)), 'std': tf.transpose(tf.sqrt(p * tf.matmul(Rsq, tf.square(inp))))}

    return self.run_on_input(get_graph(), data)

@register_method
def _get_bound_norm(self, data, ord = 2):
  """ Compute error for arbitrary norm, see Article section 2.2
      Input: data with shape (nObjects, nFeatures)
      Note that we assume error in the first layer only
  """
  self.check_p_layer0()

  @cache_graph(self)
  def get_graph(ord = ord):
    layers = self.model_correct.layers
    w_prod = np.prod([np.linalg.norm(w, ord = ord) for w in self.W[1:]])
    p = self.p_inference[1]
    return {'mean': p * w_prod * tf.norm(layers[0].input, ord = ord, axis = 1)}

  return self.run_on_input(get_graph(ord = ord), data)

# adding norm bounds
@register_method
def get_bound_v1_infnorm(self, data):
  return self._get_bound_norm(data, ord = np.inf)

@register_method
def get_bound_v1_1norm(self, data):
  return self._get_bound_norm(data, ord = 1)

@register_method
def get_bound_v1_2norm(self, data):
  return self._get_bound_norm(data, ord = 2)

@register_method
def _get_bound_sum_norm(self, ord):
  """ Calculate the norm of the weights """
  return {'mean': sum([np.linalg.norm(w, ord = ord) for w in self.W])}

# adding sum norm bounds
@register_method
def get_bound_sum_infnorm(self):
  return self._get_bound_sum_norm(ord = np.inf)
@register_method
def get_bound_sum_1norm(self):
  return self._get_bound_sum_norm(ord = 1)
@register_method
def get_bound_sum_2norm(self):
  return self._get_bound_sum_norm(ord = 2)
@register_method
def get_bound_sum_fronorm(self):
  return self._get_bound_sum_norm(ord = 'fro')
