import gin
import numpy as np

@gin.configurable
def random_p_fail(N_len, idx, p_level):
    """Failures at a random layer."""
    out = [0] * N_len
    out[idx] = p_level
    return out

@gin.configurable
def lecun_ntk_wb(n_hid_layers=5, n_units=50):
    """Get weights and biases for the case when activations have unit variance,
       input and output have dim 1."""
    N = [1] + [n_units] * n_hid_layers + [1]
    
    W = []
    B = []
    for i in range(1, len(N)):
        W.append(np.random.randn(N[i], N[i - 1]) / N[i - 1] ** 0.5)
        B.append(np.zeros(N[i]))
        
    return {'N': N, 'W': W, 'B': B}