from experiment import Experiment
import numpy as np
from helpers import *

class ConstantExperiment(Experiment):
  def __init__(self, N, P, KLips, W, B, do_print = False):
    """ Fill in the weights and initialize models """
    Experiment.__init__(self, N, P, KLips, do_print)
    
    # array with weight matrices
    self.W = W
    
    # array with biases
    self.B = B
    
    # creating "crashing" model
    self.model = create_model(self.P, self.W, self.B, self.K)
    
    # creating correct model
    self.model_no_dropout = create_model([0] * len(self.P), self.W, self.B, self.K)
    
  def get_inputs(self, how_many):
    return np.random.randn(how_many, self.N[0])
