from helpers import *
import numpy as np
from experiment import Experiment
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

    # create functions for the model
    self.create_supplementary_functions()

  def update_C(self, inputs):
    if self.activation == 'relu':
        self.C = self.mean_per_neuron(inputs)

  def reset_C(self):
    if self.activation == 'relu':
        self.C = np.zeros(np.array(self.C).shape)
    else: self.C = np.ones(np.array(self.C).shape)
    
  def get_inputs(self, how_many):
    return np.random.randn(how_many, self.N[0])
