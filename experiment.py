from helpers import *
from model import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.special import expit

# for obtaining current TF session
from keras.backend.tensorflow_backend import get_session

class Experiment():
  """ One experiment on neuron crash, contains a fixed weights network """
  def __init__(self, N, P, KLips = 1, activation = 'sigmoid', do_print = False, name = 'exp'):
    """ Initialize using given number of neurons per layer N (array), probability of failure P, and the Lipschitz coefficient """
    
    if do_print:
      print('Creating network for %d-dimensional input and %d-dimensional output, with %d hidden layers' % (N[0], N[-1], len(N) - 2))
    
    # saving N
    self.N = N

    # saving name
    self.name = name
    
    # making list if P is a number
    if type(P) == float:
      P = [P] * (len(N) - 2)
      
    # checking if the length is correct. Last and first layers cannot have failures so P is shorter than N
    assert(len(N) == len(P) + 2)
      
    # saving P, first layer has zero probability of failure
    self.P = [0.0] + P
    
    # maximal value of output from neuron (1 since using sigmoid)
    self.C = 1. if activation == 'sigmoid' else np.zeros(len(N) - 2)
    
    # saving K
    self.K = KLips

    # saving activation
    self.activation = activation

  def predict_no_dropout(self, data):
    """ Get correct network output for a given input vector """
    return self.model_no_dropout.predict(np.array([data]))[0]
  
  def predict(self, data, repetitions = 100):
    """ Get crashed network outputs for given input vector and number of repetitions """
    data = np.repeat(np.array([data]), repetitions, axis = 0)
    return self.model.predict(data)
  
  def plot_error(experiment, errors):
    """ Plot the histogram of error  """
    
    # plotting
    plt.figure()
    plt.title('Network error histogram plot')
    plt.xlabel('Network output error')
    plt.ylabel('Frequency')
    plt.hist(errors, density = True)
    #plt.plot([true, true], [0, 1], label = 'True value')
    #plt.legend()
    plt.savefig('error_hist_' + experiment.name + '.png')
    plt.show()
  
  def get_error(experiment, inp, repetitions = 100):
    """ Return error between crashed and correct networks """
    return experiment.predict(inp, repetitions = repetitions) - experiment.predict_no_dropout(inp)

  def activation_fcn(self, x):
    """ Get activation function at x """
    if self.activation == 'relu':
      x = np.copy(x)
      x[x < 0] = 0
      return self.K * x
    elif self.activation == 'sigmoid':
      return expit(4 * self.K * x)
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

  def weights_norm(self, ord = 1):
    """ Calculate the norm of the weights """
    return sum([np.linalg.norm(w, ord = ord) for w in self.W])
