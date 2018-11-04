from experiment_constant import *
from helpers import *
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
import pickle
from tqdm import tqdm
from functools import partial
import sys

class MNISTExperiment(ConstantExperiment):
  def __init__(self, N, P, KLips, epochs = 20, activation = 'sigmoid', update_C_inputs = 1000, reg_type = 0, reg_coeff = 0.01, train_dropout = None, do_print = False, scaler = 1.0, name = 'exp'):
    N = [28 ** 2] + N + [10]
#    if type(P) == list:
#        P = [0] + P + [0]
      
    """ Fill in the weights and initialize models """
   
    self.x_train, self.y_train, self.x_test, self.y_test = get_mnist(out_scaler = scaler, in_scaler = 1.)

    # dropout regularization    
    if reg_type == 'dropout':
        train_dropout = [reg_coeff] + [0] * (len(N) - 1)

    if not train_dropout:
        train_dropout = [0] * len(N)

    self.C_arr = []
    self.C_per_neuron_arr = []

    self.activation = activation

    Experiment.__init__(self, N, P, KLips, activation, do_print = False, name = name)
    model, self.reg, self.errors = create_random_weight_model(N, train_dropout, self.P, KLips, activation, reg_type = reg_type, reg_coeff = reg_coeff, C_arr = self.C_arr, C_per_neuron_arr = self.C_per_neuron_arr)
    self.model_no_dropout = model
    self.create_supplementary_functions()

    self.C_history = []

    history = []
    self.EDeltaHistory = []
    self.EDeltaV2History = []
    tqdm_ = tqdm if do_print else lambda x : x
    sys.stdout.flush()
    for i in tqdm_(range(epochs)):
        if activation == 'relu':
            self.reset_C()
            self.update_C_train(update_C_inputs)
            self.C_history += [self.C]
        # weights and biases
        self.W = model.get_weights()[0::2]
        self.B = model.get_weights()[1::2]
        self.EDeltaHistory += [self.get_mean_std_error()]
        self.EDeltaV2History += [np.mean(self.get_mean_error_v2())]
        history += [model.fit(self.x_train, self.y_train, verbose = 0, batch_size = 10000, epochs = 1, validation_data = (self.x_test, self.y_test))]

    self.reset_C()

    plt.figure()
    plt.title('Delta bound during training')
    plt.xlabel('Epoch')
    plt.ylabel('Delta bound')
    means, stds = zip(*self.EDeltaHistory)
    plt.plot(means, label = 'Mean delta')
    plt.plot(self.EDeltaV2History, label = 'Bound v2')
    means = np.array(means)
    stds = np.array(stds)
    plt.fill_between(range(len(means)), means - stds, means + stds, color = 'green', alpha = 0.2, label = 'Std delta')
    plt.legend()
    plt.savefig('delta_bound_training_' + name + '.png')
    plt.show()

    if do_print and activation == 'relu':
        plt.figure()
        plt.title('C during training')
        plt.xlabel('Epoch')
        plt.ylabel('C')
        for layer, data in enumerate(zip(*self.C_history)):
            plt.plot([np.mean(x) for x in data], label = 'Layer %d' % (layer + 1))
        plt.legend()
        plt.savefig('C_training_' + name + '.png')
        plt.show()

    val_acc = [value for epoch in history for value in epoch.history['val_acc']]
    acc = [value for epoch in history for value in epoch.history['acc']]

    if do_print:
      plt.figure()
      plt.plot(val_acc, label = 'Validation accuracy')
      plt.plot(acc, label = 'Accuracy')
      plt.legend()
      plt.savefig('accuracy_training_' + name + '.png')
      plt.show()
    
    # weights and biases
    W = model.get_weights()[0::2]
    B = model.get_weights()[1::2]

    self.original_model = model
      
    # creating "crashing" and "normal" models
    ConstantExperiment.__init__(self, N, P, KLips, W, B, activation, do_print, name = name)

  def get_accuracy(self, inputs = 1000, repetitions = 1000, tqdm_ = lambda x : x, no_dropout = False):
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
    [K.set_value(item, np.mean(value)) for item, value in zip(self.C_arr, self.C + [0])]
    [K.set_value(item, value.reshape(-1, 1)) for item, value in zip(self.C_per_neuron_arr[:-1], self.C)]
