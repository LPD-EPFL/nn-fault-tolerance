from experiment_constant import *
from helpers import *
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist

class MNISTExperiment(ConstantExperiment):
  def __init__(self, N, P, KLips, epochs = 20, activation = 'sigmoid', reg_type = 'delta', reg_coeff = 0.01, do_print = False):
    N = [28 ** 2] + N + [10]
      
    """ Fill in the weights and initialize models """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    self.x_train = np.array([elem.flatten() for elem in x_train])
    self.x_test = np.array([elem.flatten() for elem in x_test])
    self.y_train = np.array([[1 if i == digit else 0 for i in range(10)] for digit in y_train.flatten()])
    self.y_test = np.array([[1 if i == digit else 0 for i in range(10)] for digit in y_test.flatten()])
    
    model = create_random_weight_model(N, KLips, activation, reg_type = reg_type, reg_coeff = reg_coeff)
    history = model.fit(self.x_train, self.y_train, batch_size = 10000, epochs = epochs, verbose = 0, validation_data = (self.x_test, self.y_test))

    if do_print:
      plt.figure()
      plt.plot(history.history['val_acc'], label = 'Validation accuracy')
      plt.plot(history.history['acc'], label = 'Accuracy')
      plt.legend()
      plt.show()
    
    # weights and biases
    W = model.get_weights()[0::2]
    B = model.get_weights()[1::2]

    self.original_model = model
      
    # creating "crashing" and "normal" models
    ConstantExperiment.__init__(self, N, P, KLips, W, B, activation, do_print)
    
  def get_inputs(self, how_many):
    x = np.vstack((self.x_train, self.x_test))
    indices = np.random.choice(x.shape[0], how_many)
    return x[indices, :]
