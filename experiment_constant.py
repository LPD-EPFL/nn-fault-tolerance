from experiment import Experiment
import numpy as np
from helpers import *
from keras import backend as K
import helpers

class ConstantExperiment(Experiment):
  def __init__(self, N, P, KLips, W, B, activation = 'sigmoid', do_print = False, name = 'exp'):
    """ Fill in the weights and initialize models """
    Experiment.__init__(self, N, P, KLips, activation, do_print, name = name)
    
    # array with weight matrices
    self.W = W
    
    # array with biases
    self.B = B
    
    # creating "crashing" model
    self.model = create_model(self.P, self.W, self.B, self.K, activation)
    
    # creating correct model
    self.model_no_dropout = create_model([0] * len(self.P), self.W, self.B, self.K, activation)

    self.create_max_per_layer()

  def update_C(self, inputs):
    if self.activation == 'relu':
#        self.C = np.max([self.C, self.max_per_layer(inputs)], axis = 0)
        self.C = self.max_per_layer(inputs)

  def reset_C(self):
    self.C = np.zeros(np.array(self.C).shape)
    
  def get_inputs(self, how_many):
    return np.random.randn(how_many, self.N[0])
