from helpers import *
from model import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.special import expit

# for obtaining current TF session
from keras.backend.tensorflow_backend import get_session

# importing other parts of the class
import bad_input_search, bounds, process_data

# adding other parts of the class
@add_methods_from(bad_input_search, bounds, process_data)
class Experiment():
  """ One experiment on neuron crash, contains a fixed weights network """
  def __init__(self, N, W, B, p_inference, KLips = 1, activation = 'sigmoid', do_print = False, name = 'exp', check_shape = True):
    """ Initialize a crashing neurons experiment with
        N array of shapes [input, hidden_1, ..., hidden_last, output]
        W: array with matrices. The shape must be [hidden1 x input, hidden2 x hidden1, ..., output x hiddenLast]
        B: array with vectors. The shape must be [hidden1, hidden2, ..., output] 
        p_inference neuron failure probabilities, same length as N
        p_train neuron failure probabilities, same length as N
        KLips Lipschitz coefficient of the activation function
        activation Activation function type
        do_print Print a summary?
        name Name of the experiment
    """

    # printing some information    
    if do_print:
      print('Creating network for %d-dimensional input and %d-dimensional output, with %d hidden layers' % (N[0], N[-1], len(N) - 2))

    # saving check_shape argument
    self.check_shape = check_shape

    # fixing Pinference
    if p_inference == None:
      p_inference = [0] * (len(N))

    # input check
    assert len(N) == len(p_inference), "p_inference must same dimension as N"

    # saving weights/biases
    self.W = W
    self.B = B

    # creating "crashing" model
    self.model_crashing = create_fc_crashing_model(N, W, B, p_inference, KLips = KLips, func = activation, reg_spec = {}, do_print = do_print)

    # creating correct model
    self.model_correct  = create_fc_crashing_model(N, W, B, [0] * len(N), KLips = KLips, func = activation, reg_spec = {}, do_print = do_print)

    # saving N
    self.N = N

    # saving name
    self.name = name
    
    # saving p_inference
    self.p_inference = p_inference
    
    # saving K
    self.KLips = KLips

    # saving activation
    self.activation = activation

  def predict_correct(self, data):
    """ Get correct network output for a given input tensor """
    if self.check_shape:
      assert len(data.shape) == 2, "Must have input nObj x nFeatures"
      data = np.array(data).reshape(-1, self.N[0])
    return self.model_correct.predict(data)
  
  def predict_crashing(self, data, repetitions):
    """ Get crashed network outputs for given input vector and number of repetitions
        Input: array with shape (-1, dataCol)
    """
    if self.check_shape:
      assert len(data.shape) == 2, "Must have input nObj x nFeatures"
      assert data.shape[1] == self.N[0], "Input shape must be nObj x nFeatures"
      data = np.array(data).reshape(-1, self.N[0])
    data_repeated = np.repeat(data, repetitions, axis = 0)
    return self.model_crashing.predict(data_repeated).reshape(data.shape[0], repetitions, self.N[-1])
  
  def plot_error(experiment, errors):
    """ Plot the histogram of error  """
    
    plt.figure()
    plt.title('Network error histogram plot')
    plt.xlabel('Network output error')
    plt.ylabel('Frequency')
    plt.hist(errors, density = True)
    plt.savefig('error_hist_' + experiment.name + '.png')
    plt.show()
  
  def compute_error(self, data, repetitions = 100):
    """ Return error between crashed and correct networks """
    return self.predict_crashing(data, repetitions = repetitions) - np.repeat(self.predict_correct(data)[:, np.newaxis, :], repetitions, axis = 1)

  def activation_fcn(self, x):
    """ Get activation function at x """
    if self.activation == 'relu':
      x = np.copy(x)
      x[x < 0] = 0
      return self.KLips * x
    elif self.activation == 'sigmoid':
      return expit(4 * self.KLips * x)
    else: raise(NotImplementedError("Activation %s is not implemented yet" % self.activation))

  def activation_grad(self, x):
    """ Get activation function gradient at x """
    if self.activation == 'relu':
      return self.K * 1. * (x >= 0)
    elif self.activation == 'sigmoid':
      r = expit(self.K * 4 * x)
      return 4 * self.K * np.multiply(r, 1 - r)
    else: raise(NotImplementedError("Activation %s is not implemented yet" % self.activation))
 
  def forward_pass_manual(self, x):
    """ Manual forward pass (for assertions) """
    ilast = len(self.W) - 1
    for i, (w, b) in enumerate(zip(self.W, self.B)):
      x = w.T @ x + b.reshape(-1, 1)
      if i < ilast: x = self.activation_fcn(x)
    return x

  def get_activations(self, x):
    """ Get Lipschitz coefficients of act. functions per layer """
    # last layer has no activation fcn
    ilast = len(self.W) - 1

    # the resulting activations
    results = []

    # loop over layers
    for i, (w, b) in enumerate(zip(self.W, self.B)):
      # computing forward pass...
      x = w.T @ x + b.reshape(-1, 1)

      # applying activation and KLocal if there is an activation function (all but last layer)
      if i < ilast:
        Klocal = self.activation_grad(x)
        x = self.activation_fcn(x)
        results += [Klocal]

    return results

  def get_inputs(self, how_many):
    """ Get random normal input-shaped vectors """
    return np.random.randn(how_many, self.N[0])
