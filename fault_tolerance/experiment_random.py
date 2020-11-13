from helpers import *
import numpy as np
from experiment import *

class RandomExperiment(Experiment):
  """ Initialize the weights randomly """
  def __init__(self, N, mean_weight = 0.0, std_weight = 1.0, **kwargs):
    
    # array with weight matrices
    W = []
    
    # array with biases
    B = []
    
    # loop over layers
    for i in range(1, len(N)):
      # creating w and b
      w = np.random.randn(N[i], N[i - 1]) * std_weight / N[i - 1] + mean_weight
      b = np.random.randn(N[i]) / N[i] * std_weight + mean_weight
      
      # adding them to the array
      W.append(w)
      B.append(b)
   
    # initializing the base class
    Experiment.__init__(self, N, W, B, **kwargs)
