import numpy as np
from experiment_mnist import *
import pickle
experiment = MNISTExperiment([10], 0.123, 0.321, epochs = 200, activation = 'sigmoid', reg_type = 'delta',
                             reg_coeff = 0.01, do_print = True)
for a, b in zip(experiment.model_no_dropout.get_weights(), experiment.model.get_weights()):
    assert np.allclose(a, b)
weights = experiment.model_no_dropout.get_weights()
W = weights[0::2]
B = weights[1::2]
pickle.dump([W, B], open('mnistnet100percenttightness.pkl', 'wb'))
