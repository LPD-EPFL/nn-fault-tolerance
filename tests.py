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
    if activation == 'relu': exp.update_C(exp.get_inputs(1000))

    # and some input
    x0 = np.random.randn(exp.N[0], 1) * 5

    # checking that forward pass is implemented properly
    assert np.linalg.norm(exp.forward_pass_manual(x0).reshape(-1) - exp.predict_no_dropout(x0.reshape(-1))) < 1e-5, "Forward pass does not work properly: manual and TF values disagree"

    # test that norm_error(infty) == mean_v1
    assert np.allclose(exp.get_mean_std_error()[0], exp.get_norm_error(ord = np.infty)), "One of implementations of bound v1 (mean) is incorrect"

    # test that |WL|*...*|W_2|*C*p == mean_v2
    R = np.eye(exp.N[-1])
    for w in exp.W[1:][1::-1]:
        R = R @ np.abs(w.T)
    v21 = exp.get_mean_error_v2()
    inp = np.ones(exp.N[1]) if activation == 'sigmoid' else exp.C[0]
    v22 = R @ inp * max(exp.P)
    assert np.allclose(v21, v22), "One of implementations of bound v2 (mean) is incorrect got %s %s" % (str(v21), str(v22))

# testing generate_params
inp = {'a': [0, 1, 2], 'b': ['x', 'y'], 'c': [None]}
out = list(generate_params(**inp))
true_out = [{'c': None, 'b': 'x', 'a': 0},
 {'c': None, 'b': 'y', 'a': 0},
 {'c': None, 'b': 'x', 'a': 1},
 {'c': None, 'b': 'y', 'a': 1},
 {'c': None, 'b': 'x', 'a': 2},
 {'c': None, 'b': 'y', 'a': 2}]
assert out == true_out, "Generate Params must work"



print("All done")
