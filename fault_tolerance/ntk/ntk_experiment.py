from fault_tolerance.experiment import Experiment
import gin
import numpy as np
from ray import tune


@gin.configurable
def ntk_experiment(config=None, checkpoint_dir=None, n_inputs=100, n_inits=100, repetitions=1000):
    """Compute error in LeCun/NTK initialization empirically."""
    
    for init in range(n_inits):
        exp = Experiment()
        
        for inp in range(n_inputs):
            data = np.random.randn(1, 1)
            out = exp.model_correct.predict(data)
            delta = exp.compute_error(data, repetitions=repetitions)
            tune.report({'input': np.mean(data), 'out': np.mean(out),
                         'delta_mean': np.mean(delta), 'delta_std': np.std(delta),
                         'n_init': init, 'n_inp': inp})
        del exp