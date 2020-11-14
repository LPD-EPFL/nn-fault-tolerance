from fault_tolerance.experiment import Experiment
import gin
import numpy as np
from ray import tune
import tensorflow as tf
from fault_tolerance.helpers import np_random_seed


@gin.configurable
def ntk_experiment(config=None, checkpoint_dir=None, n_inputs=100, n_inits=100, repetitions=1000,
                   input_chunk_size=50):
    """Compute error in LeCun/NTK initialization empirically."""
    
    n_chunks = max(1, n_inputs // input_chunk_size)
    n_inputs = input_chunk_size * n_chunks

    with np_random_seed():
        data_all = np.random.randn(n_inputs, 1)
    
    for init in range(n_inits):
        exp = Experiment()
        print("GPUs", tf.config.experimental.list_physical_devices("GPU"))
        print("GPU", tf.test.gpu_device_name())
        
        for chunk in range(n_chunks):
            data = data_all[chunk * input_chunk_size:(chunk + 1) * input_chunk_size]
            out = exp.model_correct.predict(data)
            delta = exp.compute_error(data, repetitions=repetitions)

            for inp in range(input_chunk_size):
                tune.report({'input': data[inp, 0], 'out': out[inp, 0],
                             'delta_mean': np.mean(delta[inp]), 'delta_std': np.std(delta[inp]),
                             'n_init': init, 'n_inp': inp + chunk * input_chunk_size,
                             'inp_chunk': chunk})
        del exp