from experiment import Experiment
import numpy as np
from helpers import *

class RandomExperiment(Experiment):
  def __init__(self, N, P, KLips, do_print = False):
    """ Fill in the weights and initialize models """
    Experiment.__init__(self, N, P, KLips, do_print)
    
    # array with weight matrices
    self.W = []
    
    # array with biases
    self.B = []
    
    # loop over layers
    for i in range(1, len(self.N)):
      # creating w and b
      w = np.random.randn(self.N[i - 1], self.N[i]) / (self.N[i - 1]) * 5
      b = np.random.randn(self.N[i]) / self.N[i] ** 2
      
      # adding them to the array
      self.W.append(w)
      self.B.append(b)
      
    # creating "crashing" model
    self.model = create_model(self.P, self.W, self.B, self.K)
    
    # creating correct model
    self.model_no_dropout = create_model([0] * len(self.P), self.W, self.B, self.K)
    
  def get_inputs(self, how_many):
    return np.random.randn(how_many, self.N[0])
