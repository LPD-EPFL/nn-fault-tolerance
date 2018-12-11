from keras import backend as K
from helpers import *
import numpy as np
from experiment import *
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
import sys

class TrainExperiment(Experiment):
  def __init__(self, x_train, y_train, x_test, y_test, N, Pinference = None, Ptrain = None, task = 'classification', KLips = 1, epochs = 20, activation = 'sigmoid', reg_type = 0, reg_coeff = 0.01, do_print = False, name = 'exp'):
    """ Get a trained with MSE loss network with configuration (N, P, activation) and reg_type(reg_coeff) with name. The last layer is linear
        N: array with shapes [hidden1, hidden2, ..., hiddenLast]. Input and output shapes are determined automatically
        Pinference: array with [p_input, p_h1, ..., p_hlast, p_output]: inference failure probabilities
        Ptrain: same for the train
    """

    # fixing Pinference
    if Pinference == None:
      Pinference = [0] * (len(N) + 2)

    # fixing Ptrain
    if Ptrain == None:
      Ptrain = [0] * (len(N) + 2)

    # obtaining input/output shape
    input_shape = x_train[0].size
    output_shape = y_train[0].size

    # full array of shapes
    N = [input_shape] + N + [output_shape]

    # input check
    assert task in ['classification', 'regression'], "Only support regression and classification"
    assert len(Pinference) == len(Ptrain), "Pinference and Ptrain must have the same length"
    assert len(N) == len(Ptrain), "Ptrain must have two more elements compared to N"
    assert input_shape > 0, "Input must exist"
    assert output_shape > 0, "Output must exist"

    # filling in the task
    self.task = task

    # remembering the dataset
    self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    # creating weight initialization
    W, B = [], []
    for i in range(1, len(N)):
      W += [np.random.randn(N[i], N[i - 1]) * np.sqrt(2. / N[i - 1]) / KLips]
      B += [np.random.randn(N[i])]

    # creating a model
    model = create_fc_crashing_model(N, W, B, Ptrain, KLips = KLips, func = activation, reg_type = reg_type, reg_coeff = reg_coeff, do_print = do_print)

    # fitting the model on the train data
    history = model.fit(x_train, y_train, verbose = do_print, batch_size = 10000, epochs = epochs, validation_data = (x_test, y_test))

    # plotting the loss
    if do_print:
      plt.figure()

      # determining what to plot (target)
      if task == 'classification':
        target = 'acc'
      elif task == 'regression':
        target = 'loss'
      else: raise NotImplementedError("Plotting for this task is not supported")

      # plotting
      plt.plot(history.history['val_' + target], label = 'val_' + target)
      plt.plot(history.history[target], label = target)
      plt.legend()
      plt.savefig('training_' + name + '.png')
      plt.show()
    
    # obtaining trained weights and biases
    W = model.get_weights()[0::2]
    B = model.get_weights()[1::2]

    # creating "crashing" and "normal" models
    ConstantExperiment.__init__(self, N, P, KLips, W, B, activation, do_print, name = name)
    self.layers = self.model_no_dropout.layers[:-1]

  def get_accuracy(self, inputs = 1000, repetitions = 1000, tqdm_ = lambda x : x, no_dropout = False):
    if self.task != 'classify':
      print("Warning: the task is not a classification task")
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
    """ Get random inputs from the dataset. If how_many = 'all', then compute on all """
    x = np.vstack((self.x_train, self.x_test))
    if how_many == 'all': return x
    indices = np.random.choice(x.shape[0], how_many, replace = False)
    return x[indices, :]
