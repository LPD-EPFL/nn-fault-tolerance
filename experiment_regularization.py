from helpers import *
from matplotlib import pyplot as plt
import numpy as np
from experiment_datasets import *
import pickle

class MNISTExperimentRegularized(TrainExperiment):
    def __init__(self, N, p_inference = None, p_train = None, KLips = 1, epochs = 30, activation = 'sigmoid',
                 reg_type = None, reg_coeff = 0.01, do_print = False, name = 'exp', seed = 0,
                 batch_size = 10000, reg_bound = None, reg_bound_coeff = 0.0):
        """ Get a trained with MSE loss network with configuration (N, P, activation) and reg_type(reg_coeff) with name. The last layer is linear
                N: array with shapes [hidden1, hidden2, ..., hiddenLast]. Input and output shapes are determined automatically
                p_inference: array with [p_input, p_h1, ..., p_hlast, p_output]: inference failure probabilities
                Ptrain: same for the train
                Regularizes the network with the v3 bound
                Set reg_bound to v2/v3/v4 string and reg_bound_coeff to some value
        """
        
        # remembering the dataset
        self.x_train, self.y_train, self.x_test, self.y_test = get_mnist(out_max = 1.0, in_max = 1.0)
        x_train, y_train, x_test, y_test = self.x_train, self.y_train, self.x_test, self.y_test

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
        assert len(p_inference) == len(p_train), "Pinference and p_train must have the same length"
        assert len(N) == len(p_train), "Ptrain must have two more elements compared to N"
        assert input_shape > 0, "Input must exist"
        assert output_shape > 0, "Output must exist"

        # filling in the task
        self.task = 'classification'
        
        # seeding the weights generation
        np.random.seed(seed)

        # creating weight initialization
        W, B = [], []
        for i in range(1, len(N)):
            W += [np.random.randn(N[i], N[i - 1]) * np.sqrt(2. / N[i - 1]) / KLips]
            B += [np.random.randn(N[i])]

        # print?
        do_print_ = True if do_print == True else False
    
        # by default, no regularization with a bound
        bound_loss = tf.Variable(0.0)
    
        # creating a model
        model, parameters = create_fc_crashing_model(N, W, B, p_train, KLips = KLips, func = activation,
                                         reg_type = reg_type, reg_coeff = reg_coeff, do_print = do_print_,
                                         do_compile = False)
        
        # saving parameters to allow for the bound method to create the graph...
        self.p_inference = p_inference
        self.N = N
        self.model_correct = model
    
        # name of the graph cache attribute
        bound_cache = None
        
        # if bound type is set...
        if reg_bound:
            # obtaining the bound method by name
            bound_method = getattr(self, 'get_bound_%s' % reg_bound)

            # calling on some dummy data to create the graph
            bound_method(np.random.randn(1, N[0]))
            
            # obtaining the std bound graph
            bound_cache = '__cache_get_bound_%s_get_graph_args_()_kwargs_{}' % reg_bound
            bound_loss = getattr(self, bound_cache)['std'] ** 2
            
            # regularization = coeff * mean over the dataset
            bound_loss = tf.reduce_mean(bound_loss) * reg_bound_coeff
        
        # obtaining its original loss
        orig_loss = parameters['loss']
        
        # adding bound loss to parameter
        parameters['loss'] = lambda y_true, y_pred: orig_loss(y_true, y_pred) + bound_loss
        
        # compiling the model
        model.compile(**parameters)
    
        # fitting the model on the train data
        history = model.fit(x_train, y_train, verbose = do_print_, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test))
    
        # plotting the loss
        if do_print and epochs > 0:
    
          # determining what to plot (target)
          if self.task == 'classification':
            target = 'categorical_accuracy'
          else: raise NotImplementedError("Plotting for this task is not supported")
    
          # plotting
          plt.figure()
          plt.plot(history.history['val_' + target], label = 'val_' + target)
          plt.plot(history.history[target], label = target)
          plt.legend()
          plt.savefig('training_' + name + '.png')
          plt.show()
        
          # plotting
          plt.figure()
          plt.plot(history.history['loss'], label = 'loss')
          plt.legend()
          plt.show()
    
        # obtaining trained weights and biases
        W = model.get_weights()[0::2]
        W = [w.T for w in W]
        B = model.get_weights()[1::2]
    
        if bound_cache:
            # clearning bound cache
            delattr(self, bound_cache)
    
        # creating "crashing" and "normal" models
        Experiment.__init__(self, N, W, B, p_inference, KLips = KLips, activation = activation, do_print = do_print_, name = name)
