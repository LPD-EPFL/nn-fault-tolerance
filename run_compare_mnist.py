import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# getting proc ID
import sys
nProc = int(sys.argv[1])
worker = int(sys.argv[2])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from time import time, sleep

from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
import pickle
from helpers import *
from experiment_mnist import *

# Parameters: layers, pfail, Lipschitz, repetitions
N = [200, 100, 80, 50, 20]
P = [0.5,  0,  0,  0,  0]
KLips = 1.
repetitions = list(range(1500))

def get_point(name = ''):
  """ Perform an experiment once: train the network and get its error """
  experiment = MNISTExperiment(N, P, KLips, epochs = 200, activation = 'relu', do_print = True, reg_type = 0, scaler = 1.0, name = name)
  experiment.create_supplementary_functions()
  experiment.update_C(experiment.get_inputs(10000))
  point = {}
  point['mean_std'] = experiment.get_mean_std_error()
  point['mean_v2'] = experiment.get_mean_error_v2()
  point['run'] = experiment.run(repetitions = 5000, inputs = 100)
  point['W'] = experiment.W
  point['B'] = experiment.B
  return point

if worker == 0:
  os.system("telegram-send 'N %s P %s Repetitions %d'" % (N, P, len(repetitions)))

# repetitions for me
repetitions_me = repetitions[worker::nProc]

# repetitions for me
total = len(repetitions_me)

# results
results = []

# starting time
tstart = time()
trained = 0
for k, repetition in enumerate(repetitions_me):
  results += [get_point('rep%03d' % repetition)]
  #sleep(5)

  # saving data at each iteration to prevent data loss
  pickle.dump(results, open('results_%d.pkl' % worker, 'wb'))

  # calculating and sending progress
  trained += 1
  delta_sec = int((time() - tstart) / trained * (total - trained))
  eta_h = delta_sec // 3600
  eta_m = (delta_sec - eta_h * 3600) // 60
  eta_s = delta_sec - eta_h * 3600 - eta_m * 60
  os.system("telegram-send 'Process %d/%d progress %d/%d ETA %02d:%02d:%02d'" % (worker + 1, nProc, trained, total, eta_h, eta_m, eta_s))

os.system("telegram-send 'Process %d/%d done'" % (worker + 1, nProc))


