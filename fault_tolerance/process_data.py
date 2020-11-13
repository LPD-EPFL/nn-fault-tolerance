from helpers import *
import numpy as np
from tqdm import tqdm
import matplotlib
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
  data_points = r[main_key].shape[0]

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

  font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

  matplotlib.rc('font', **font)

  # plotting bound comparison, if requested
  if do_plot:
    plt.figure()
    #plt.title('Bound comparison ' + name)
    plt.ylabel('Absolute correlation with experiment')
    plt.bar(other_keys, np.abs([corr[x] for x in other_keys]))
    plt.xticks(rotation=70)
    plt.savefig('figures/comparison_corr_%s.pdf' % name, bbox_inches = 'tight')
    plt.show()

  # plotting bound comparison via rank loss
  if do_plot:
    plt.figure(figsize=(3,2))
    #plt.title('Bound comparison ' + name)
    plt.ylabel('Rank loss')
    other_byloss = sorted(other_keys, key = lambda x : loss[x])
    plt.bar(other_keys, [loss[x] for x in other_byloss])
    plt.xticks(rotation=85)
    plt.savefig('figures/comparison_rank_%s.pdf' % name, bbox_inches = 'tight')
    plt.show()

  # plot subplots, name_fcn(key, idx) -> str; fcn(axis, key)
  def key_subplots(name_fcn, fcn):
    fig, axs = plt.subplots(3, 2, figsize=(10, 13))
    axs = axs.ravel()
    for i, key in enumerate(other_keys):
      if i > 5: break
      axs[i].set_title(name_fcn(key, i))
      if i + 1 == len(other_keys):
        axs[i].set_xlabel(main_key)
      axs[i].set_ylabel(key)
      fcn(axs[i], key)
    plt.show()
    return fig

  # plotting error of error histogram
  if do_plot:
    fig = key_subplots(lambda key, i: 'Error of error' if i == 0 else 'e.e.',
                       lambda ax, key : ax.hist(data[main_key] - data[key]))
    fig.savefig('figures/comparison_ee_%s.pdf' % name, bbox_inches = 'tight')

  # plotting absolute relative error of error
  if do_plot:
    def prepare_data(key):
      rel_error = np.abs((data[main_key] - data[key]) / data[main_key])
      rel_error[rel_error >= 2] = 2
      return rel_error
    fig = key_subplots(lambda key, i: 'Abs. rel. error of error min\'d w 2' if i == 0 else 'min(2,a.r.e.e.)',
                       lambda ax, key : ax.hist(prepare_data(key)))
    fig.savefig('figures/comparison_aree_%s.pdf' % name, bbox_inches = 'tight')

  # plotting boxplot of absolute relative error of error
  if do_plot:
    def prepare_data(key, cutoff = 2):
      rel_error = np.abs((data[main_key] - data[key]) / (1e-20 + data[main_key]))
      rel_error[rel_error >= cutoff] = cutoff
      return rel_error
    plt.figure(figsize=(3,2))
    plt.boxplot([prepare_data(key) * 100 for key in other_keys], labels = other_keys)
    plt.ylabel('Relative error, %')
    plt.xticks(rotation=85)
    plt.savefig('figures/comparison_boxplot_aree_%s.pdf' % name, bbox_inches = 'tight')

  # plotting scatter plots with experimental mean, if requested
  if do_plot:
    fig = key_subplots(lambda key, i: '%s=%.2f Loss=%.2f' % (name + ', corr with exp' if i == 0 else 'c.w.e.', corr[key], loss[key]),
                       lambda ax, key: ax.scatter(data[main_key], data[key] * np.sign(corr[key])))
    fig.savefig('figures/comparison_scatter_%s.pdf' % name, bbox_inches = 'tight')

  # returning comparison dataframe
  return res
