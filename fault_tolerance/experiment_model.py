from model import *
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
        
        # creating duplicate input
        inp = Input(shape = in_shape)

        # correct model: the original model with additional layer (crashes in that layer == crashes in input)
        self.model_correct  = Model(inputs = inp, outputs = model(IdentityLayer     (   input_shape = in_shape)(inp)))

        # crashing model: with independent crashes
        self.model_crashing = Model(inputs = inp, outputs = model(IndependentCrashes(p, input_shape = in_shape)(inp)))
        
        # disable bounds input shape check (experimental!)
        self.check_shape = False

class ModelCrashExperiment(Experiment):
    def __init__(self, model, p_inference=None, p = None, name='exp'):
        """ Get an experiment based on a model, assuming failures everywhere
            p failures everywhere (constant probability)
            p_inference array with crash probabilities, must match model.layers in size
        """

        # saving p_inference
        assert p_inference is None or p is None, "Cannot have both p and p_inference set"
        assert p_inference is not None or p is not None, "Cannot have both p and p_inference not set"

        # if no p_inference but p present, set to always p
        if p is not None: p_inference = [p] * len(model.layers)

        # saving p_inference
        self.p_inference = p_inference
       
        # obtaining input/output shape 
        out_shape = model.layers[-1].output.shape[1:]

        # sanity check
        assert len(out_shape) == 1, "Only support 1D output"

        # only need the last component of output
        self.N = [-1, int(out_shape[0])]
        
        # correct model: the original model with additional layer (crashes in that layer == crashes in input)
        self.model_correct  = faulty_model(model, [0] * len(model.layers))

        # crashing model: with independent crashes
        self.model_crashing = faulty_model(model, self.p_inference)
        
        # disable bounds input shape check (experimental!)
        self.check_shape = False
