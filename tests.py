from experiment_random import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# some pfail
p = 1e-2

# loop over activation fcns
for activation in ['sigmoid', 'relu']:
    # creating an experiment...
    exp = RandomExperiment([50, 40, 30, 5], [p, 0], 1, activation = activation)

    # and some input
    x0 = np.random.randn(exp.N[0], 1) * 5

    # checking that forward pass is implemented properly
    assert np.linalg.norm(exp.forward_pass_manual(x0).reshape(-1) - exp.predict_no_dropout(x0.reshape(-1))) < 1e-5, "Forward pass does not work properly: manual and TF values disagree"

print("All done")
