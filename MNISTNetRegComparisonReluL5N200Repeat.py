# setting up GPU to not consume all memory
# using GPU 0
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# getting proc ID
import sys
nProc = int(sys.argv[1])
worker = int(sys.argv[2])

# standard data science imports
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from experiment_mnist import *
from tfshow import *
import pickle

# experiment parameters

# number of neurons/layer
N = [200, 150, 100, 50, 20]
Layers = len(N)

# Lipsch. coefficient
KLips = 1

# activation function
activation = 'relu'

# output scaler
scaler = 1.0

# epochs to train
epochs = 100

# parameters for accuract calculation
inputs = 1000
acc_param = 1000

# p failure at first level
pfirst_options = [0.06]#np.linspace(0, 0.2, 6)[1:-1]

# regularization types
reg_type_options = ['delta', 'delta_network', 'l1', 'l2', 0]

# regularization coeffs
reg_coeff_options = [0] + list(np.logspace(-10, 0, 6))[:-1]

# repetitions for all processes
repetitions = list(range(12))

# repetitions for this process
repetitions_ = repetitions[worker::nProc]

def get_results(pfirst = 0.5, reg_type = 'delta', reg_coeff = 1e-4, repetition = 0):
    """ Run a single experiment """

    # skipping if nontrivial coefficient specified for no regularization
    if reg_type == 0 and reg_coeff != 0:
        return {}

    # printing the parameters
    print('Parameters', pfirst, reg_type, reg_coeff, repetition)
    P = [pfirst] + [0] * (Layers - 1)

    # name for the images
    name = 'pfirst_%s_reg_type_%s_coeff_%s_repetition_%s' % (str(pfirst), str(reg_type), str(reg_coeff), str(repetition))

    # creating the experiment with parameters
    model = MNISTExperiment(N, P, KLips, epochs = epochs, activation = activation, reg_type = reg_type,
                            reg_coeff = reg_coeff, do_print = True, scaler = scaler, name = name)

    # set max per layer
    model.update_C(model.get_inputs(10000))

    # header for the experiment
    header = ['mean_exp', 'std_exp', 'mean_bound', 'std_bound', 'output_mean', 'output_variance', 'mean_bound_v2']

    # running the experiment and obtaining the bound
    bound = model.run(inputs = inputs, repetitions = 1000)

    # get accuracy w. dropout
    acc = model.get_accuracy(acc_param, acc_param, tqdm_ = tqdm)

    # get accuracy without dropout
    acc_orig = model.get_accuracy(acc_param, acc_param, tqdm_ = tqdm, no_dropout = True)

    # results
    results = {x: y for x, y in zip(header, bound)}

    # adding accuracy to the results
    results['acc_dropout'] = acc
    results['acc_orig'] = acc_orig

    # printing data
    print('ACC', acc, acc_orig)

    # saving weights (just in case)
    results['W'] = model.W
    results['B'] = model.B

    # Important: freeing up the memory
    K.clear_session()

    # result
    return results

# printing info
print('P', pfirst_options)
print('N', N)
print('Reg', reg_type_options)
print('Coeff', reg_coeff_options)
print('Repetitions', repetitions_)
print('Need to run: %d' % (len(pfirst_options) * len(repetitions_) * ((len(reg_type_options)) * (len(reg_coeff_options)) + 1)))

# saving info
results['info'] = (pfirst_options, reg_type_options, reg_coeff_options, repetitions, N, Layers, KLips, activation, scaler, epochs, inputs, acc_param)

# RUNNING the experiment
results = {(pfirst, reg_type, reg_coeff, repetition): get_results(pfirst = pfirst, reg_type = reg_type, reg_coeff = reg_coeff, repetition = repetition)
 for repetition in repetitions_
 for reg_coeff in reg_coeff_options
 for reg_type in reg_type_options
 for pfirst in pfirst_options
}

# saving data
pickle.dump(results, open('results_repeat%d.pkl' % worker, 'wb'))
