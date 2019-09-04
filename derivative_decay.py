# Helpers to measure derivative decay on MNIST

from experiment_train import *
from experiment_datasets import *

def get_exp(epochs = 15, N = [200, 50], reg_coeff = 0.0005, reg_type = 'continuous', do_print = 'plot'):
    """ Train and return the result """
    def get_p_arr(p):
        """ p array with failure on the first layer """
        return [0, p]  + [0] * len(N)

    # Lips. coeff
    KLips = 1.

    # activation function
    activation = 'sigmoid'

    # training the network
    exp = MNISTExperiment(N = N, p_inference = get_p_arr(0), p_train = get_p_arr(0),
                          KLips = KLips, epochs = epochs,
                          activation = activation, reg_type = reg_type,
                          reg_coeff = reg_coeff, do_print = do_print,
                          name = 'experiment_weights', seed = None, batch_size = 1000)
    
    # returning the weights in the middle
    return exp

def parameters_for_N(N, parameters):
    """ Parameters for get_exp for a particular N """
    param1 = {x: y for x, y in parameters.items()}
    param1['N'] = [N, 100]
    return param1

def experiment_for_N(N, parameters):
    """ Experiment with N """
    return get_exp(**parameters_for_N(N, parameters))

def mean_dLdy(exp, irange = range(60000)):
    """ Mean derivative w.r.t. the input for the model """
    # obtain the loss
    loss = exp.model_correct.loss(exp.output_tensor, exp.model_correct.output)

    # obtain first layer output
    y1 = exp.model_correct.layers[0].output

    # input tensor
    inp = exp.model_correct.input

    # output tensor with answers
    out = exp.output_tensor

    # gradients w.r.t. first layer output
    dL_dy1 = tf.gradients(loss, y1)[0]

    # some input/output pairs
    x = exp.x_train[irange]
    y = exp.y_train[irange]

    # mean over everything, should decay as 1/n_l
    dL_dy1_abs_mean_B_mean_N = tf.reduce_mean(tf.reduce_mean(tf.abs(dL_dy1), axis = 0))

    sess = get_session()

    return {'D': sess.run(dL_dy1_abs_mean_B_mean_N, feed_dict = {inp: x, out: y})}

def mean_d2Ldy2(exp, irange = range(10)):
    """ Mean hessian w.r.t. the input for the model """
    # obtain the loss
    loss = exp.model_correct.loss(exp.output_tensor, exp.model_correct.output)

    # obtain first layer output
    y1 = exp.model_correct.layers[0].output

    # input tensor
    inp = exp.model_correct.input

    # output tensor with answers
    out = exp.output_tensor

    # some input/output pairs
    x = exp.x_train[irange]
    y = exp.y_train[irange]

    # obtaining a session
    sess = get_session()

    # mean over inputs
    mean_loss = tf.reduce_mean(loss)

    # d^2Loss/dy
    #H = tf.reduce_mean(tf.hessians(mean_loss, y1)[0], axis = [0, 2])
    H = tf.hessians(mean_loss, y1)[0]
    H = tf.reduce_mean(tf.linalg.diag_part(tf.transpose(H, (1, 3, 0, 2))), axis = 2)

    # computing the hessian
    Hnp = sess.run(H, feed_dict = {inp: x, out: y})

    # computing mean for diag/all hessian
    H_diag_mean = np.mean(np.abs(np.diag(Hnp)))
    H_all_mean = np.mean(np.abs(Hnp.flatten()))

    return {'H_diag': H_diag_mean, 'H_all': H_all_mean}

def get_metrics(exp, to_run):
    """ Compute all metrics for an experiment, return a dict """
    result = {}
    for fcn in to_run:
        result.update(fcn(exp))
    return result

def line_1_bias(x, C, coeff = -1):
    """ Line passing (0,0) with a fixed coeff """
    return coeff * x + C

def get_arr(key, results):
    """ Get results for a key """
    return [[x[key] for x in y] for y in results]
