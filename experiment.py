from helpers import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.special import expit

# for obtaining current TF session
from keras.backend.tensorflow_backend import get_session

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

  def get_exact_std_error_v3_better(self, x, output_tensor = False):
    """ Exact error up to O(p^2) in case even if x_i are not small """
    if type(x) == list:
       x = np.array(x).reshape(1, -1)
    elif len(x.shape) == 1:
       x = x.reshape(1, -1)

    layers = self.model_no_dropout.layers
    first_layer = layers[0]
    second_layer = layers[1]
    last_layer = layers[-1]

    # need to drop all components one by one in the second layer input
    N2 = int(second_layer.input.shape[1])
    outputs = []
    for i in range(N2):
      l1out = first_layer.output
      mask = [0 if i == j else 1 for j in range(N2)]
    
      # dropped data
      y = tf.multiply(l1out, mask)
    
      # implementing the rest of the network
      for layer in layers[1:]:
        y = layer.activation(tf.matmul(y, layer.weights[0]) + layer.weights[1])
      outputs.append(tf.square(y - last_layer.output))
    res = max(self.P) * sum(outputs)
    res = tf.sqrt(res)
    if not output_tensor:
      res = get_session().run(res, feed_dict = {first_layer.input.name: x})
    return res

  def get_exact_error_v3_better(self, x, output_tensor = False):
    """ Exact error up to O(p^2) in case even if x_i are not small """
    if type(x) == list:
       x = np.array(x).reshape(1, -1)
    elif len(x.shape) == 1:
       x = x.reshape(1, -1)

    layers = self.model_no_dropout.layers
    first_layer = layers[0]
    second_layer = layers[1]
    last_layer = layers[-1]

    # need to drop all components one by one in the second layer input
    N2 = int(second_layer.input.shape[1])
    outputs = []
    for i in range(N2):
      l1out = first_layer.output
      mask = [0 if i == j else 1 for j in range(N2)]
    
      # dropped data
      y = tf.multiply(l1out, mask)
    
      # implementing the rest of the network
      for layer in layers[1:]:
        y = layer.activation(tf.matmul(y, layer.weights[0]) + layer.weights[1])
      outputs.append(y - last_layer.output)
    res = max(self.P) * sum(outputs)
    if not output_tensor:
      res = get_session().run(res, feed_dict = {first_layer.input.name: x})
    return res

  def get_exact_error_v3_tf(self, x):
    """ Same as get_exact_error_v3 but uses TF implementation """
    if type(x) == list:
       x = np.array(x).reshape(1, -1)
    elif len(x.shape) == 1:
       x = x.reshape(1, -1)

    # resulting gradient w.r.t. first layer output
    grad = []

    # list of layers
    layers = self.model_no_dropout.layers

    # for all output dimensions
    for output_dim in range(self.N[-1]):
      # get derivative of output
      out = layers[-1].output[:, output_dim]

      # w.r.t. first layer output
      grad += [tf.reduce_sum(tf.multiply(tf.gradients([out], [layers[0].output])[0], layers[0].output), axis = 1)]

    # comput the result
    res = -max(self.P) * np.array(get_session().run(grad, feed_dict = {layers[0].input.name: x})).T
    return res

  def get_exact_std_error_v3_tf(self, x):
    """ Calculate std on an input """
    if type(x) == list:
       x = np.array(x).reshape(1, -1)
    elif len(x.shape) == 1:
       x = x.reshape(1, -1)

    # resulting gradient w.r.t. first layer output
    grad = []

    # list of layers
    layers = self.model_no_dropout.layers

    # for all output dimensions
    for output_dim in range(self.N[-1]):
      # get derivative of output
      out = layers[-1].output[:, output_dim]

      # w.r.t. first layer output
      grad += [tf.reduce_sum(tf.multiply(tf.square(tf.gradients([out], [layers[0].output])[0]), tf.square(layers[0].output)), axis = 1)]

    # comput the result
    res = max(self.P) * np.array(get_session().run(grad, feed_dict = {layers[0].input.name: x})).T
    res = np.sqrt(res)
    return res

  def get_exact_std_error_v3(self, x, ifail = 0):
    """ Exact error std for a given input. ifail = 0 for first layer or -1 for failing input """

    if not hasattr(self, 'get_exact_std_error_v3_complained'):
        print("This function does not result in true variance: self.get_exact_std_error_v3")
    self.get_exact_std_error_v3_complained = True

    # reshaping to a column vector if given a list or vector
    if type(x) == list:
       x = np.array(x).reshape(-1, 1)
    elif len(x.shape) == 1:
       x = x.reshape(-1, 1)

    # last layer has no activation fcn
    ilast = len(self.W) - 1

    # probability of failure
    p = max(self.P)

    # the error (will be redefined if ifail >= 0)
    error = p * np.square(x)

    # loop over layers
    for i, (w, b) in enumerate(zip(self.W, self.B)):
      # computing forward pass...
      x = w.T @ x + b.reshape(-1, 1)

      # obtaining local Lipschitz coefficient (the derivative)
      Klocal = np.ones(x.shape)

      # applying activation and KLocal if there is an activation function (all but last layer)
      if i < ilast:
        Klocal = self.activation_grad(x)
        x = self.activation_fcn(x)

      # at the failing layer, copying output...
      if i == ifail:
        error = p * np.square(x)

      # at later stages propagating the failure
      elif i > ifail:
        # multiplying by the weight matrix
        error = np.square(w.T) @ error

        # checking that can multiply element-wise with Klocal (Lipschitz coeffs)
        assert Klocal.shape == error.shape, "Shapes Klocal=%s error=%s must agree" % (str(Klocal.shape), str(error.shape))

        # multiplying by the activation
        error = np.multiply(np.square(Klocal), error)

    # return std
    return np.sqrt(error)

  def get_exact_error_v3(self, x, ifail = 0):
    """ Exact error for a given input. ifail = 0 for first layer or -1 for failing input """
    # reshaping to a column vector if given a list or vector
    if type(x) == list:
       x = np.array(x).reshape(-1, 1)
    elif len(x.shape) == 1:
       x = x.reshape(-1, 1)

    # last layer has no activation fcn
    ilast = len(self.W) - 1

    # probability of failure
    p = max(self.P)

    # the error (will be redefined if ifail >= 0)
    error = -p * np.copy(x)

    # loop over layers
    for i, (w, b) in enumerate(zip(self.W, self.B)):
      # computing forward pass...
      x = w.T @ x + b.reshape(-1, 1)

      # obtaining local Lipschitz coefficient (the derivative)
      Klocal = np.ones(x.shape)

      # applying activation and KLocal if there is an activation function (all but last layer)
      if i < ilast:
        Klocal = self.activation_grad(x)
        x = self.activation_fcn(x)

      # at the failing layer, copying output...
      if i == ifail:
        error = -p * np.copy(x)

      # at later stages propagating the failure
      elif i > ifail:
        # multiplying by the weight matrix
        error = w.T @ error

        # checking that can multiply element-wise with Klocal (Lipschitz coeffs)
        assert Klocal.shape == error.shape, "Shapes Klocal=%s error=%s must agree" % (str(Klocal.shape), str(error.shape))

        # multiplying by the activation
        error = np.multiply(Klocal, error)

    # return signed error
    return error

  def get_norm_error(self, ord = 2, do_abs = False):
    """ Compute error for arbitrary norm, see Article section 2.2 """
    C = None
    if self.activation == 'sigmoid':
      C = self.C
    elif self.activation == 'relu':
      C = np.linalg.norm(self.C[0], ord = ord)
    else: raise(NotImplementedError("Function does not work with activation " + self.activation))
    res = max(self.P) * C
    for w in self.W[1:]:
      w1 = np.abs(w) if do_abs else w
      res *= np.linalg.norm(w1.T, ord = ord)
    return res

  def get_mean_error_v3(self, data, do_abs = False):
    # computing total weight matrix
    R = np.eye(self.N[-1])
    for w in self.W[::-1]:
      w1 = np.abs(w) if do_abs else w
      R = R @ w1.T

    # computing errors
    err = [R @ inp for inp in data]

    # returning mean error
    return -max(self.P) * np.array(err)
  
  def run(self, repetitions = 10000, inputs = 50, do_plot = True, do_print = True, do_tqdm = True, randn = None, inputs_update = None):
    """ Run a single experiment with a fixed network """

    # Creating input data
    if type(inputs) == int:
        data = self.get_inputs(inputs)
        # no inputs -> no output
        if inputs == 0: return None
    else: data = inputs

    # computing max value per neuron for ReLU
    if self.activation == 'relu':
        self.update_C(data)
        if randn:
            self.update_C(np.random.randn(randn, self.N[0]))
        if inputs_update:
            self.update_C(self.get_inputs(inputs_update))


    # computing error v3
    mean_v3_approx = self.get_mean_error_v3(data)
    mean_v3_exact = self.get_exact_error_v3(np.array(data).T).T
    mean_v2 = self.get_mean_error_v2()
    std_v3_better = self.get_exact_std_error_v3_better(data)
    mean_v3_better = self.get_exact_error_v3_better(data)
    std_v3_exact = self.get_exact_std_error_v3_tf(data)
    std_v3_square = self.get_exact_std_error_v3(data)

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

    # get activations
    # todo: make it one forward pass for each x
    activations = [[np.mean(y) for y in self.get_activations(x)] for x in data]

    # product of the norms of matrices
    norm_l1 = self.get_norm_error(ord = 1)
    norm_l2 = self.get_norm_error(ord = 2)

    # sum of the norms of matrices
    norm_s_l1 = self.weights_norm(ord = 1)
    norm_s_l2 = self.weights_norm(ord = 2)
    norm_s_F = self.weights_norm(ord = 'fro')

    # Printing results summary
    if do_print:
      print('Absolute Error; average over inputs, average over dropout:')
      print('True values array mean: %f std %f' % (np.mean(trues), np.std(trues)))
      print('Bound L1Prod  %f' % np.max(np.abs(norm_l1)))
      print('Bound L2Prod  %f' % np.max(np.abs(norm_l2)))
      print('Bound L1Sum   %f' % np.max(np.abs(norm_s_l1)))
      print('Bound L2Sum   %f' % np.max(np.abs(norm_s_l2)))
      print('Bound FSum    %f' % np.max(np.abs(norm_s_F)))
      print('Bound v1      %f Std %f' % (mean_bound, std_bound))
      print('Bound v2      %f' % np.mean(mean_v2))
      print('Bound v3 app  %f' % np.max(np.abs(mean_v3_approx)))
      print('Bound v3 exct %f Std %f' % (np.max(np.abs(mean_v3_exact)), np.max(np.abs(std_v3_exact))))
      print('Bound v3 bttr %f Std %f' % (np.max(np.abs(mean_v3_better)), np.max(np.abs(std_v3_better))))
      print('Experiment    %f Std %f' % (mean_exp, std_exp))
      print('MeanAct %s' % str(np.mean(activations, axis = 0)))

    # Returning summary
    return {
     'input': data,
     'activations': activations,
     'output': trues,
     'error_exp_mean': np.mean(errors, axis = 1),
     'error_exp_std': np.std(errors, axis = 1),
     'error_abs_exp_mean': np.mean(errors_abs, axis = 1),
     'error_abs_exp_std': np.std(errors_abs, axis = 1),
     'error_v1_mean': mean_bound,
     'error_v1_std': std_bound,
     'error_v2_mean': mean_v2,
     'error_v3_mean_approx': mean_v3_approx,
     'error_v3_mean_exact': mean_v3_exact,
     'error_v3_mean_better': mean_v3_better,
     'error_v3_std_exact': std_v3_exact,
     'error_v3_std_better': std_v3_better,
     'error_v3_std_square': std_v3_square,
     'error_matnorm_prod_l1': norm_l1,
     'error_matnorm_prod_l2': norm_l2,
     'error_matnorm_sum_l1': norm_s_l1,
     'error_matnorm_sum_l2': norm_s_l2,
     'error_matnorm_sum_F': norm_s_F,
    }

  def weights_norm(self, ord = 1):
    """ Calculate the norm of the weights """
    return sum([np.linalg.norm(w, ord = ord) for w in self.W])

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
