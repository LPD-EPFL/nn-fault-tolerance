from experiment_constant import *
from helpers import *
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.datasets import mnist
import pickle
from tqdm import tqdm
from functools import partial

class MNISTExperiment(ConstantExperiment):
  def __init__(self, N, P, KLips, epochs = 20, activation = 'sigmoid', update_C_inputs = 1000, reg_type = 0, reg_coeff = 0.01, train_dropout = None, do_print = False):
    N = [28 ** 2] + N + [10]
      
    """ Fill in the weights and initialize models """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    self.x_train = np.array([elem.flatten() for elem in x_train])
    self.x_test = np.array([elem.flatten() for elem in x_test])
    self.y_train = np.array([[1 if i == digit else 0 for i in range(10)] for digit in y_train.flatten()])
    self.y_test = np.array([[1 if i == digit else 0 for i in range(10)] for digit in y_test.flatten()])
    
    if not train_dropout:
        train_dropout = [0] * len(N)

    self.C_arr = []

    self.activation = activation

    model, self.reg = create_random_weight_model(N, train_dropout, KLips, activation, reg_type = reg_type, reg_coeff = reg_coeff, C_arr = self.C_arr)
    self.model_no_dropout = model
    Experiment.__init__(self, N, P, KLips, activation, do_print = False)
    self.create_max_per_layer()

    self.C_history = []

    tqdm_ = tqdm if do_print else lambda x : x
    history = []
    for i in tqdm_(range(epochs)):
        if activation == 'relu':
            self.reset_C()
            self.update_C_train(update_C_inputs)
            self.C_history += [self.C]
        history += [model.fit(self.x_train, self.y_train, verbose = 0, batch_size = 10000, epochs = 1, validation_data = (self.x_test, self.y_test))]

    if do_print and activation == 'relu':
        plt.figure()
        plt.title('C during training')
        plt.xlabel('Epoch')
        plt.ylabel('Error per layer')
        for layer, data in enumerate(zip(*self.C_history)):
            plt.plot(data, label = 'Layer %d' % (layer + 1))
        plt.legend()
        plt.show()

    val_acc = [value for epoch in history for value in epoch.history['val_acc']]
    acc = [value for epoch in history for value in epoch.history['acc']]

    if do_print:
      plt.figure()
      plt.plot(val_acc, label = 'Validation accuracy')
      plt.plot(acc, label = 'Accuracy')
      plt.legend()
      plt.show()
    
    # weights and biases
    W = model.get_weights()[0::2]
    B = model.get_weights()[1::2]

    self.original_model = model
      
    # creating "crashing" and "normal" models
    ConstantExperiment.__init__(self, N, P, KLips, W, B, activation, do_print)
  def get_accuracy(self, inputs = 10000, repetitions = 10000, tqdm_ = lambda x : x, no_dropout = False):
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
  def get_inputs(self, how_many):
    x = np.vstack((self.x_train, self.x_test))
    indices = np.random.choice(x.shape[0], how_many)
    return x[indices, :]
  def update_C_train(self, inputs):
    self.update_C(self.get_inputs(inputs))
    [K.set_value(item, value) for item, value in zip(self.C_arr, self.C + [0])]
