from experiment import *
from keras import Model, Input
from keras.layers import Lambda

class ModelInputCrashExperiment(Experiment):
    def __init__(self, model, p=0.01, name='exp'):
        """ Get an experiment based on a model, assuming input failures
            p failures at input
        """

        # saving p_inference
        self.p_inference = [0, p]
       
        # obtaining input/output shape 
        in_shape = model.layers[0].input.shape[1:]
        out_shape = model.layers[-1].output.shape[1:]

        # sanity check
        assert len(out_shape) == 1, "Only support 1D output"

        # only need the last component of output
        self.N = [-1, int(out_shape[0])]
        
        def IdentityLayer(input_shape=None):
            """ A layer which does nothing """
            return Lambda(
                lambda x: x + 0, input_shape=input_shape, name='Identity')

        # creating duplicate input
        inp = Input(shape = in_shape)

        # correct model: the original model with additional layer (crashes in that layer == crashes in input)
        self.model_correct  = Model(inputs = inp, outputs = model(IdentityLayer     (   input_shape = in_shape)(inp)))

        # crashing model: with independent crashes
        self.model_crashing = Model(inputs = inp, outputs = model(IndependentCrashes(p, input_shape = in_shape)(inp)))
        
        # disable bounds input shape check (experimental!)
        self.check_shape = False
