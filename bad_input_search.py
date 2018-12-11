from helpers import *

# for adding functions to Experiment class
__methods__ = []
register_method = register_method(__methods__)

@register_method
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
