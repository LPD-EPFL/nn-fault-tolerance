from experiment_constant import ConstantExperiment
import numpy as np
from helpers import *

class RandomExperiment(ConstantExperiment):
  def __init__(self, N, P, KLips, activation = 'sigmoid', do_print = False):
    
    # array with weight matrices
    W = []
    
    # array with biases
    B = []
    
    # loop over layers
    for i in range(1, len(N)):
      # creating w and b
      w = np.random.randn(N[i - 1], N[i]) / (N[i - 1]) * 5 + 1
      b = np.random.randn(N[i]) / N[i]
      
      # adding them to the array
      W.append(w)
      B.append(b)
   
    ConstantExperiment.__init__(self, N, P, KLips, W, B, activation, do_print)
    
  def get_inputs(self, how_many):
    return np.random.randn(how_many, self.N[0])
