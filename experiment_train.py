from keras import backend as K
from helpers import *
import numpy as np
from experiment_constant import *
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from functools import partial
import sys

class TrainExperiment(ConstantExperiment):
  def __init__(self, x_train, y_train, x_test, y_test, N, P, task = 'classification', KLips = 1, epochs = 20, activation = 'sigmoid', reg_type = 0, reg_coeff = 0.01, do_print = False, name = 'exp'):
    """ Get a trained with MSE loss network with configuration (N, P, activation) and reg_type(reg_coeff) with name. The last layer is linear
        N: array with shapes [hidden1, hidden2, ..., hiddenLast]. Input and output shapes are determined automatically
        Pinference: array with [p_input, p_h1, ..., p_hlast, p_output]: inference failure probabilities
        Ptrain: same for the train
    """

    # input check
    assert task in ['classification', 'regression'], "Only support regression and classification"

    # obtaining input/output shape
    input_shape = x_train[0].size
    output_shape = y_train[0].size
    N = [input_shape] + N + [output_shape]

    self.classify = classify

#    if type(P) == list:
#        P = [0] + P + [0]
      
    """ Fill in the weights and initialize models """
   
    self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    train_dropout = [0] * len(N)

    self.activation = activation

    Experiment.__init__(self, N, P, KLips, activation, do_print = False, name = name)
    model, self.reg, self.errors = create_random_weight_model(N, train_dropout, self.P, KLips, activation, reg_type = reg_type, reg_coeff = reg_coeff, train_dropout_l1 = train_dropout_l1)
    self.model_no_dropout = model
    self.layers = model.layers[:-1]
    if reg_type == 'dropout':
        self.layers = self.layers[:1] + self.layers[2:]
    self.create_supplementary_functions()

    history = model.fit(self.x_train, self.y_train, verbose = do_print, batch_size = 10000, epochs = epochs, validation_data = (self.x_test, self.y_test))

    if do_print:
      plt.figure()
      if classify:
        plt.plot(history.history['val_acc'], label = 'Validation accuracy')
        plt.plot(history.history['acc'], label = 'Accuracy')
      else:
        plt.plot(history.history['val_loss'], label = 'Validation loss')
        plt.plot(history.history['loss'], label = 'Loss')
      plt.legend()
      plt.savefig('training_' + name + '.png')
      plt.show()
    
    # weights and biases
    W = model.get_weights()[0::2]
    B = model.get_weights()[1::2]

    self.original_model = model
      
    # creating "crashing" and "normal" models
    ConstantExperiment.__init__(self, N, P, KLips, W, B, activation, do_print, name = name)
    self.layers = self.model_no_dropout.layers[:-1]

  def get_accuracy(self, inputs = 1000, repetitions = 1000, tqdm_ = lambda x : x, no_dropout = False):
    if not self.classify:
      print("Warning: the task is a regression task")
    if no_dropout: repetitions = 1
    x = np.vstack((self.x_train, self.x_test))
    y = np.vstack((self.y_train, self.y_test))
    indices = np.random.choice(x.shape[0], inputs)
    data = x[indices, :]
    answers = np.argmax(y[indices], axis = 1)
    predict_method = (lambda x : np.argmax(self.predict_no_dropout(x))) if no_dropout else (lambda x : np.argmax(self.predict(x, repetitions = repetitions), axis = 1))
    predictions = [predict_method(inp) for inp in tqdm_(data)]
    correct = [pred == ans for pred, ans in zip(predictions, answers)]
    return np.sum(correct) / (inputs * repetitions)
  def get_mae_nocrash(self):
     """ Get mean absolute error for train and test datasets """
    err_train = np.mean(np.abs(self.model_no_dropout.predict(self.x_train) - self.y_train))
    err_test  = np.mean(np.abs(self.model_no_dropout.predict(self.x_test)  - self.y_test))
    return {'train': err_train, 'test': err_test}
  def get_inputs(self, how_many):
    x = np.vstack((self.x_train, self.x_test))
    indices = np.random.choice(x.shape[0], how_many, replace = False)
    return x[indices, :]
