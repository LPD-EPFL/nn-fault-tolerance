from helpers import *
import numpy as np
from experiment_train import *
from keras.datasets import mnist, boston_housing, fashion_mnist

def get_mnist(out_max = 1.0, in_max = 255., fashion = False):
    """ Get the MNIST dataset with input/output max values as x_train, y_train, x_test, y_test """
    dataset = fashion_mnist if fashion else mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = x_train.reshape(-1, 28 ** 2) / 255. * in_max
    x_test = x_test.reshape(-1, 28 ** 2) / 255. * in_max
    digits = {x: [out_max if y == x else 0. for y in range(10)] for x in range(10)}
    y_train = np.array([digits[y] for y in y_train])
    y_test = np.array([digits[y] for y in y_test])
    return x_train, y_train, x_test, y_test

class BostonHousingExperiment(TrainExperiment):
  """ Create the Boston Housing experiment """
  def __init__(self, **kwargs):
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    TrainExperiment.__init__(self, x_train, y_train, x_test, y_test, task = 'regression', **kwargs)

class MNISTExperiment(TrainExperiment):
  """ Create the MNIST dataset experiment """
  def __init__(self, **kwargs):
    x_train, y_train, x_test, y_test = get_mnist(out_max = 1.0, in_max = 1.0)
    TrainExperiment.__init__(self, x_train, y_train, x_test, y_test, task = 'classification', **kwargs)

class FashionMNISTExperiment(TrainExperiment):
  """ Create the Fashion MNIST dataset experiment """
  def __init__(self, **kwargs):
    x_train, y_train, x_test, y_test = get_mnist(out_max = 1.0, in_max = 1.0, fashion = True)
    TrainExperiment.__init__(self, x_train, y_train, x_test, y_test, task = 'classification', **kwargs)
