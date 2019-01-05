from keras import backend as K
from helpers import *
from experiment import *
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
import sys

class TrainExperiment(Experiment):
  def __init__(self, x_train, y_train, x_test, y_test, N, p_inference = None, p_train = None, task = 'classification', KLips = 1, epochs = 20, activation = 'sigmoid', reg_type = None, reg_coeff = 0.01, do_print = False, name = 'exp', seed = 0, batch_size = 10000):
    """ Get a trained with MSE loss network with configuration (N, P, activation) and reg_type(reg_coeff) with name. The last layer is linear
        N: array with shapes [hidden1, hidden2, ..., hiddenLast]. Input and output shapes are determined automatically
        p_inference: array with [p_input, p_h1, ..., p_hlast, p_output]: inference failure probabilities
        Ptrain: same for the train
    """

    # fixing Pinference
    if p_inference == None:
      p_inference = [0] * (len(N) + 2)

    # fixing Ptrain
    if p_train == None:
      p_train = [0] * (len(N) + 2)

    # obtaining input/output shape
    input_shape = x_train[0].size
    output_shape = y_train[0].size

    # full array of shapes
    N = [input_shape] + N + [output_shape]

    # input check
    assert task in ['classification', 'regression'], "Only support regression and classification"
    assert len(p_inference) == len(p_train), "Pinference and p_train must have the same length"
    assert len(N) == len(p_train), "Ptrain must have two more elements compared to N"
    assert input_shape > 0, "Input must exist"
    assert output_shape > 0, "Output must exist"

    # filling in the task
    self.task = task

    # remembering the dataset
    self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
    if len(self.y_train.shape) == 1:
      self.y_train = self.y_train.reshape(-1, 1)
      self.y_test  = self.y_test.reshape(-1, 1)

    # seeding the weights generation
    np.random.seed(seed)

    # creating weight initialization
    W, B = [], []
    for i in range(1, len(N)):
      W += [np.random.randn(N[i], N[i - 1]) * np.sqrt(2. / N[i - 1]) / KLips]
      B += [np.random.randn(N[i])]

    # print?
    do_print_ = True if do_print == True else False

    # creating a model
    model = create_fc_crashing_model(N, W, B, p_train, KLips = KLips, func = activation, reg_type = reg_type, reg_coeff = reg_coeff, do_print = do_print_)

    # fitting the model on the train data
    history = model.fit(x_train, y_train, verbose = do_print_, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test))

    # plotting the loss
    if do_print and epochs > 0:
      plt.figure()

      # determining what to plot (target)
      if task == 'classification':
        target = 'categorical_accuracy'
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
    W = [w.T for w in W]
    B = model.get_weights()[1::2]

    # creating "crashing" and "normal" models
    Experiment.__init__(self, N, W, B, p_inference, KLips = KLips, activation = activation, do_print = do_print_, name = name)

    # sanity check
    for i, (w_new, w_old) in enumerate(zip(self.model_correct.get_weights(), model.get_weights())):
      diff = np.abs(w_new - w_old)
      max_diff = matrix_argmax(diff)
      assert np.allclose(w_new, w_old), "Error setting the weights %d %s %s %f" % (i, str(w_new.shape), str(w_old.shape), diff[max_diff])

  def get_accuracy_correct(self):
    if self.task != 'classification':
      print("Warning: the task is not a classification task")

    acc_test  = argmax_accuracy(self.predict_correct(self.x_test) , self.y_test)
    acc_train = argmax_accuracy(self.predict_correct(self.x_train), self.y_train)
    return {'train': acc_train, 'test': acc_test}

  def get_accuracy_crash(self, repetitions = 10):
    if self.task != 'classification':
      print("Warning: the task is not a classification task")

    def _get_accuracy(x, y):
      """ Get crashing accuracy as a mean over repetitions and objects of whether or not the class was correct """

      # get predicted values, argmax over output dimension
      y_pred = np.argmax(self.predict_crashing(x, repetitions = repetitions), axis = 2)

      # get true values, repeat to match the shape of y_pred
      y_true = np.repeat(np.argmax(y, axis = 1)[:, np.newaxis], repetitions, axis = 1)

      # accuracy = mean number of true predictions
      return np.mean(y_pred == y_true)

    return {'train': _get_accuracy(self.x_train, self.y_train), 'test': _get_accuracy(self.x_test, self.y_test)}

  def get_mae_crash(self, repetitions = 10):
    err_test  = np.mean(np.abs(self.predict_crashing(self.x_test , repetitions = repetitions) - np.repeat(self.y_test[:,  np.newaxis, :], repetitions, axis = 1)))
    err_train = np.mean(np.abs(self.predict_crashing(self.x_train, repetitions = repetitions) - np.repeat(self.y_train[:, np.newaxis, :], repetitions, axis = 1)))
    return {'train': err_train, 'test': err_test}

  def get_mae_correct(self):
    """ Get mean absolute error for train and test datasets """
    err_train = np.mean(np.abs(self.predict_correct(self.x_train) - self.y_train))
    err_test  = np.mean(np.abs(self.predict_correct(self.x_test)  - self.y_test))
    return {'train': err_train, 'test': err_test}

  def get_inputs(self, how_many):
    """ Get random inputs from the dataset. If how_many = 'all', then compute on all """
    x = np.vstack((self.x_train, self.x_test))
    if how_many == 'all': return x
    indices = np.random.choice(x.shape[0], how_many, replace = False)
    return x[indices, :]
