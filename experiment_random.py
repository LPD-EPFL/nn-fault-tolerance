from experiment_constant import ConstantExperiment
import numpy as np
from helpers import *

class RandomExperiment(ConstantExperiment):
  def __init__(self, N, P, KLips, activation = 'sigmoid', do_print = False, mean_weight = 0, std_weight = 1):
    
    # array with weight matrices
    W = []
    
    # array with biases
    B = []
    
    # loop over layers
    for i in range(1, len(N)):
      # creating w and b
      w = np.random.randn(N[i - 1], N[i]) / (N[i - 1]) * std_weight + mean_weight
      b = np.random.randn(N[i]) / N[i] * std_weight + mean_weight
      
      # adding them to the array
      W.append(w)
      B.append(b)
   
    ConstantExperiment.__init__(self, N, P, KLips, W, B, activation, do_print)
    
  def get_inputs(self, how_many):
    return np.random.randn(how_many, self.N[0])
