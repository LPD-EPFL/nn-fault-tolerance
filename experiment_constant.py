from experiment import Experiment
import numpy as np
from helpers import *

class ConstantExperiment(Experiment):
  def __init__(self, N, P, KLips, W, B, activation = 'sigmoid', do_print = False):
    """ Fill in the weights and initialize models """
    Experiment.__init__(self, N, P, KLips, activation, do_print)
    
    # array with weight matrices
    self.W = W
    
    # array with biases
    self.B = B
    
    # creating "crashing" model
    self.model = create_model(self.P, self.W, self.B, self.K, activation)
    
    # creating correct model
    self.model_no_dropout = create_model([0] * len(self.P), self.W, self.B, self.K, activation)

    # output for each layer https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    model = self.model_no_dropout
    inp = model.input
    outputs = [K.max(K.abs(layer.output)) for layer in model.layers[:-1]] # max over inputs over neurons
    max_per_layer = K.function([inp, K.learning_phase()], outputs)
    self.max_per_layer = lambda x : max_per_layer([x, 1])

  def update_C(self, inputs):
    if self.activation == 'relu':
        self.C = np.max([self.C, self.max_per_layer(inputs)], axis = 0)
    
  def get_inputs(self, how_many):
    return np.random.randn(how_many, self.N[0])
