from helpers import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys

# for adding functions to Experiment class
__methods__ = []
register_method = register_method(__methods__)

# All functions assume only crashes at first layer

@register_method
def preprocess_scalar_input_scalar_output(self, r):
  assert all([np.array(x).shape in [(), (1,), (1, 1)] for x in r[0]]), "Inputs must be scalar"

  # list of dicts -> dict of arrays
  r = {key: np.array([values[key] for values in r]).flatten() for key in r[0]}

  return r

@register_method
def process_scalar_output(self, r, name = "", do_plot = True):
  """ Process results (mean/std) from self.run() """
  assert self.N[-1] == 1, "Must have scalar output to compare all bounds"

  # experimental error data
  main_key = 'experiment'

  # number of data points
  data_points = r['experiment'].shape[0]

  # all compatible bounds
  all_keys = [key for key, value in r.items() if np.array(value).shape in [(data_points,), (data_points, 1)]]

  # dict with data columns
  data = {key: np.array(r[key]).flatten() for key in all_keys}

  # correlation coefficient
  corr = pd.DataFrame(data).corr()[main_key]

  # resorting keys
  all_keys = sorted(all_keys, key = lambda x : -abs(corr[x]))

  # all keys but main
  other_keys = [x for x in all_keys if x != main_key]

  # rank loss
  loss = {x: min(y, 1 - y) for x, y in compute_rank_losses(data, main_key).items()}

  # resulting comparison dataframe
  res = pd.DataFrame({'bound': other_keys, 'corr': [corr[x] for x in other_keys], 'rank_loss': [loss[x] for x in other_keys]})

  # plotting bound comparison, if requested
  if do_plot:
    plt.figure()
    plt.title('Bound comparison ' + name)
    plt.ylabel('Absolute correlation with experiment')
    plt.bar(other_keys, np.abs([corr[x] for x in other_keys]))
    plt.xticks(rotation=70)
    plt.show()

  # plotting scatter plots with experimental mean, if requested
  if do_plot:
    fig, axs = plt.subplots(3, 2, figsize=(10, 13))
    axs = axs.ravel()
    for i, key in enumerate(other_keys):
      if i > 5: break
      axs[i].set_title('%s=%.2f Loss=%.2f' % (name + ', corr. with exp.' if i == 0 else 'C', corr[key], loss[key]))
      if i + 1 == len(other_keys):
        axs[i].set_xlabel(main_key)
      axs[i].set_ylabel(key)
      axs[i].scatter(data[main_key], data[key] * np.sign(corr[key]))
    plt.show()

  # returning comparison dataframe
  return res
