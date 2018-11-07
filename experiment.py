from helpers import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class Experiment():
  """ One experiment on neuron crash, contains a fixed weights network """
  def __init__(self, N, P, KLips, activation = 'sigmoid', do_print = False, name = 'exp'):
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

  def create_supplementary_functions(self):
    # output for each layer https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    model = self.model_no_dropout
    inp = model.input
    outputs = [K.mean(K.abs(layer.output), axis = 0) for layer in model.layers[:-1]] # mean over inputs, per-neuron
    self.mean_per_neuron = lambda x : (K.function([inp, K.learning_phase()], outputs))([x, 1])
    
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
  
  def get_wb(self, layer):
    """ Get weight and bias matrix """
    return np.vstack((self.W[layer], self.B[layer]))
  
  def get_max_f(self, layer, func):
    """ Maximize func(weights) over neurons in layer """
    wb = self.W[layer]
    res = [func(w_neuron) for w_neuron in wb.T]
    return np.max(res)
  
  def get_all_f(self, layer, func):
    """ Return array func(weights) over neurons in layer """
    wb = self.W[layer]
    #print(wb.shape, len(wb.T))
    res = [func(w_neuron) for w_neuron in wb.T]
    return res

  def get_max_f_xy(self, layer, func, same_only = False):
    """ Maximize func(w1, w2) over neurons in layer """
    wb = self.W[layer]
    if same_only: res = [func(w_neuron, w_neuron) for w_neuron in wb.T]
    else: res = [func(w_neuron1, w_neuron2) for w_neuron1 in wb.T for w_neuron2 in wb.T]
    return np.max(res)
 
  def get_C(self, layer):
   return self.C if self.activation == 'sigmoid' else np.max(self.C[layer - 1])

  def get_Carr(self, layer):
   return np.ones(self.N[layer]) if self.activation == 'sigmoid' else self.C[layer - 1]

  def get_mean_std_error(self):
    """ Get theoretical bound for mean and std of error given weights """

    # Expectation of error
    EDelta = 0.

    # Expectation of error squared
    EDelta2 = 0.

    # Array of expectations
    EDeltaArr = [0]

    # Array of expectations of squares
    EDelta2Arr = [0]

    # Loop over layers
    for layer in range(1, len(self.W)):
      is_last = layer + 1 == len(self.W)

      # probability of failure of a single neuron
      p_l = self.P[layer]

      C = self.get_C(layer)

      # maximal 1-norm of weights
      w_1_norm = self.get_max_f(layer, norm1)

      # alpha from article for layer
      alpha = self.get_max_f_xy(layer, dot_abs, same_only = is_last)

      # beta from article for layer
      beta = self.get_max_f_xy(layer, norm1_minus_dot_abs, same_only = is_last)

      # a, b from article for EDelta2 (note that old EDelta is used)
      a = C ** 2 * p_l * (alpha + p_l * beta) + 2 * self.K * C * p_l * (1 - p_l) * beta * EDelta
      b = self.K ** 2 * (1 - p_l) * (alpha + (1 - p_l) * beta)

      # Updating EDelta2
      EDelta2 = a + b * EDelta2

      # Updating EDelta
      EDelta = p_l * w_1_norm * C + self.K * w_1_norm * (1 - p_l) * EDelta

      # Adding new values to arrays
      EDeltaArr.append(EDelta)
      EDelta2Arr.append(EDelta2)

    # Debug output
    #print(EDeltaArr)
    #print(EDelta2Arr)
    self.EDeltaArr = EDeltaArr

    # Returning mean and sqrt(std^2)
    return EDelta, EDelta2 ** 0.5

  def get_mean_error_v2(self):
    """ Get theoretical bound for mean error given weights, the improved version """

    # Expectation of error
    EDelta = np.zeros(self.N[1])

    # Array of expectations
    EDeltaArr = [EDelta]

    # Loop over layers
    for layer in range(1, len(self.W)):
      is_last = layer + 1 == len(self.W)

      # probability of failure of a single neuron
      p_l = self.P[layer]

      # array of max output per neuron for layer
      Carr = self.get_Carr(layer)

      # Updating EDelta: getting the weight matrix
      W1 = np.array(self.get_all_f(layer, np.abs))

      # Weight matrix x Max_per_layer vector
      Wc = W1 @ Carr

      # Updating delta evector
      EDelta = p_l * Wc + self.K * (1 - p_l) * (W1 @ EDelta)

      # Adding new values to array
      EDeltaArr.append(EDelta)

    # Debug output
    #print(EDeltaArr)
    self.EDeltaArr = EDeltaArr

    # Returning mean and sqrt(std^2)
    return EDelta
  
  def run(self, repetitions = 10000, inputs = 50, do_plot = True, do_print = True, do_tqdm = True, randn = None, inputs_update = None):
    """ Run a single experiment with a fixed network """

    # Creating input data
    data = self.get_inputs(inputs)

    if self.activation == 'relu':
        self.update_C(data)
        if randn:
            self.update_C(np.random.randn(randn, self.N[0]))
        if inputs_update:
            self.update_C(self.get_inputs(inputs_update))

    if inputs == 0: return (0, 0, 0, 0, 0)

    # Computing true values
    trues = [self.predict_no_dropout(value) for value in data]

    # Running the experiment
    tqdm_ = tqdm if do_tqdm else (lambda x : x)
    errors = [self.get_error(value, repetitions = repetitions) for value in tqdm_(data)]
 
    # Computing Maximal Absolute Mean/Std Error over 
    errors_abs = np.abs(errors)
    means = np.mean(errors_abs, axis = 1)
    stds = np.std(errors_abs, axis = 1)
    mean_exp = np.max(means)
    std_exp = np.max(stds)

    # Computing bound values
    mean_bound, std_bound = self.get_mean_std_error()

    # Plotting the error histogram
    if do_plot:
      self.plot_error(np.array(errors).reshape(-1))

    # Printing results summary
    if do_print:
      print('Error; maximal over inputs, average over dropout:')
      print('True values array mean: %f variance %f' % (np.mean(trues), np.std(trues)))
      print('Experiment %f Std %f' % (mean_exp, std_exp))
      print('Equation   %f Std %f' % (mean_bound, std_bound))
      print('Tightness  %.1f%% Std %.1f%%' % (100 * mean_exp / mean_bound, 100 * std_exp / std_bound))

    # Returning summary
    return mean_exp, std_exp, mean_bound, std_bound, np.mean(trues), np.std(trues), self.get_mean_error_v2()

  def bad_input_search(self, random_seed = 42, repetitions = 1000, to_add = 20, to_keep = 5, maxiter = 20, scaler = 1, use_std = False):
    # Trying genetic search for x
    np.random.seed(random_seed)

    mean_bound, std_bound = self.get_mean_std_error()

    bound = mean_bound
    func = np.mean
    title = 'Mean'
    if use_std:
        bound = std_bound
        func = np.std
        title = 'Std'

    # Setting parameters and creating the experiment
    N = self.N[0]

    # creating initial inputs
    data = np.random.randn(to_keep, N)

    # percents from theoretical bound
    percents = []

    for _ in range(maxiter):
      data_ = data

      # Randomizing inputs
      for input_ in data_:
        rand_direction = np.random.randn(to_add, N)
        #rand_direction /= np.linalg.norm(rand_direction, axis = 0)
        rand_direction *= scaler
        input_ = input_ + rand_direction
        data = np.vstack((data, input_))

      # Computing true values
      trues = [self.predict_no_dropout(value) for value in data]
    
      # Running the experiment
      errors = [self.get_error(value, repetitions = repetitions) for value in data]

      if self.activation == 'relu':
          self.update_C(data)

      mean_bound, std_bound = self.get_mean_std_error()

      bound = mean_bound
      if use_std:
        bound = std_bound

      # List of errors for inputs
      error_array = func(np.max(np.abs(errors), axis = 2), axis = 1)

      max_exp = np.max(error_array)
  
      # Choosing maximal error
      indices = np.argsort(-error_array)

      # Choosing best to_keep entries
      data = data[indices[:to_keep]]
  
      percent = 100 * max_exp / mean_bound
      print(title + ' error %.5f, %.2f%% from theoretical, norm %.2f' % (max_exp, percent, np.linalg.norm(data.flatten()) / data.shape[0]))
      percents.append(percent)
 
    plt.figure() 
    plt.xlabel('Iteration count')
    plt.ylabel('Percent from theoretical bound')
    plt.plot(percents)
    plt.show()

    return data[0]
