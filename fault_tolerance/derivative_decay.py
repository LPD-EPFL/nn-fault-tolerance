# Helpers to measure derivative decay on MNIST

from experiment_train import *
from experiment_datasets import *
from scipy.optimize import curve_fit

def get_exp(epochs = 15, N = [200, 50], do_print = 'plot', experiment = MNISTExperiment, p = 0.0, **kwargs):
    """ Train and return the result """
    def get_p_arr(p):
        """ p array with failure on the first layer """
        return [0, p]  + [0] * len(N)

    # Lips. coeff
    KLips = 1.

    # activation function
    activation = 'sigmoid'

    # training the network
    exp = experiment(N = N, p_inference = get_p_arr(p), p_train = get_p_arr(0),
                          KLips = KLips, epochs = epochs,
                          activation = activation, do_print = do_print,
                          name = 'experiment_weights', seed = None, batch_size = 1000, **kwargs)
    
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

def get_activation_profile(exp, layer = 0, input_idx = 0, plot = True):
    """ Get a (hopefully) smooth activation profile """

    # obtaining the model
    m = exp.model_correct

    # obtaining the tf session
    sess = get_session()

    #print(exp.history.history)

    # some input
    x = exp.x_train[input_idx:input_idx + 1]
    #x = np.random.randn(1, *exp.x_train[0].shape)

    # obtaining the activation profile
    out1 = sess.run(m.layers[layer].output, feed_dict = {m.input: x})

    if plot:
        # plotting the activation profile
        plt.figure(figsize=(16, 3))
        plt.plot(out1[0])
        plt.ylabel('$y_i$')
        plt.xlabel('$i_%d=1..n_%d$' % (layer + 1, layer + 1))
        plt.ylim((0,1))
        plt.show()

    # showing the regularization output
    #print("int [W'_{t_1}(t_1,t_0)]=%.2f" % sess.run(Continuous()(exp.W[0].T)))

    return out1[0]

def show_W_profile(exp, layer = 0):
    """ Get a W[1] """
    
    sess = get_session()
    
    def process_layer(layer):
        W_T = exp.W[layer].T
        profile = get_activation_profile(exp, layer = layer, plot = True)
        cont_tensor = Continuous()(W_T, return_dict = True) 
        #print(W_T.shape, cont_tensor)
        print("IntDer, Conv", sess.run(cont_tensor))
        return profile
    
    results = {}

    for layer in range(len(exp.W)):
        print("Layer", layer)
        results['act_%d' % layer] = process_layer(layer)

    # true answer
    print('True ans #0', np.argmax(exp.y_train[0]))

    return results

def show_neurons_1(exp):
    # showing neurons at first layer weight patterns
    # we see that neurons are grouped!
    neurons_1 = [int(t) for t in np.linspace(0, exp.W[0].shape[0] - 1, 16)]

    fig, axs = plt.subplots(1, len(neurons_1), figsize=(16, 3), sharex='row', sharey = 'row')

    for i, n in enumerate(neurons_1):
        axs[i].imshow(exp.W[0][n,:].reshape(28, 28), cmap = 'gray')
    plt.show()

    return {}
            
def W_inf_norm(exp):
    """ Returns Inf-norms for weight matrices, to show continuous
    limit (there they must stay ~constant and not blow up/decay) """
    
    # list of matrices
    Ws = exp.W
    
    # norms for everything
    Wnorms = {'W_%d' % i: np.linalg.norm(W.T, ord = 1) for i, W in enumerate(Ws)}
    
    Wnorms['W_prod'] = np.prod([x for x in Wnorms.values()])
    
    return Wnorms

def dataset_metrics(exp):
    """ Experiment -> dict of metrics (acc/loss) """
    result = {'val_acc': exp.history.history['val_categorical_accuracy'][-1],
     'train_acc':  exp.history.history['categorical_accuracy'][-1],
     'val_loss': exp.history.history['val_loss'][-1],
     'train_loss': exp.history.history['loss'][-1]}
    print(result)
    return result

def get_results(Ns, repetitions, parameters, to_run):
    """ Run get_exp for each of Ns, with repeat, with parameters, measure each from to_run, return 2D array of dicts """
    results = []
    for N in tqdm(Ns):
        # results for one N, many repetitions
        buffer = []
        for rep in range(repetitions):
            exp = experiment_for_N(N, parameters)
            buffer.append(get_metrics(exp, to_run))
            tf.reset_default_graph()
            K.clear_session()
        results.append(buffer)
    return results

def plot_results(Ns, results, name = 'decay_some'):
    # slopes will be plotted in log scale and with estimated decay rate
    slopes = {'D': -1, 'H_all': -2, 'H_diag': -2, 'var_delta': -1}

    # keys to ignore (activations are vectors)
    ignore_keys = ['act_']

    # key being processed
    for key in results[0][0].keys():

        # ignoring keys
        if any([key.startswith(k) for k in ignore_keys]):
            continue

        # obtaining the data
        xs = Ns
        ys = get_arr(key, results)

        # one of metrics for which we need a slope -> log scale + fitting a curve
        if key in slopes:
            xs = np.log(xs)
            ys = np.log(ys)

        # mean/std
        ys_mean = np.mean(ys, axis = 1)
        ys_std = np.std(ys, axis = 1)

        # plotting mean/std
        plt.figure()
        plt.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha = 0.3, color = 'green')
        plt.scatter(xs, ys_mean, color = 'green')

        # one of metrics for which we need a slope -> log scale + fitting a curve
        if key in slopes:
            # get desired slope (-1/-2)
            slope = slopes[key]

            # fitting the curve
#            fcn = partial(line_1_bias, coeff = slope)
            fcn = line_1_bias

            C = curve_fit(fcn, xs, ys_mean)[0]

            plt.plot(xs, fcn(xs, *C), color = 'red')

            plt.xlabel('log(n)')
            plt.ylabel('log(%s) slope=%.2f' % (key, C[1]))
        else:
            plt.xlabel('n')
            plt.ylabel('%s' % key)

        plt.title('%s' % (key))
        plt.savefig('figures/%s_%s.pdf' % (name, key), bbox_inches = 'tight')
        plt.show()
