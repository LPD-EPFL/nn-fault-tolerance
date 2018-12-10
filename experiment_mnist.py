from helpers import *
from experiment_train import *

class MNISTExperiment(TrainExperiment):
  def __init__(self, scaler = 1.0, **kwargs):
    x_train, y_train, x_test, y_test = get_mnist(out_scaler = scaler, in_scaler = 1.)
    TrainExperiment.__init__(self, x_train, y_train, x_test, y_test, **kwargs)
