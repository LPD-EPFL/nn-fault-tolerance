# standard imports
import numpy as np
from functools import partial

# calculate first norm
norm1 = partial(np.linalg.norm, ord = 1)

# calculate second norm
norm2 = partial(np.linalg.norm, ord = 2)

def dot_abs(x, y):
  """ Dot product between absolute values of vectors x, y """
  return np.dot(np.abs(x), np.abs(y))

def norm1_minus_dot_abs(x, y):
  """ Product of first norms - dot product between absolute values """
  return norm1(x) * norm1(y) - dot_abs(x, y)

def generate_params(**kwargs):
    """ Arguments -> array of dicts """
    
    # fetched the last parameter
    if len(kwargs) == 0:
        yield {}
        return
    
    # some argument
    param = list(kwargs.keys())[0]
    
    # the rest of the dictionary
    kwargs1 = {x: y for x, y in kwargs.items() if x != param}
    
    # loop over kwargs data
    for val in kwargs[param]:
        # loop over experiments
        for res in generate_params(**kwargs1):
            res[param] = val
            yield {x: y for x, y in res.items()}

def rank_loss(a, b):
    """ For given a, b compute the average number of misordered pairs, O(n^2) """
    
    # flattening data
    a, b = np.array(a).flatten(), np.array(b).flatten()
    
    # checking shape
    assert len(a) == len(b), "Lengths must agree"
    
    # sorting b in order of a
    b = np.array(b)[np.argsort(a)]
    
    # number of bad pairs
    res = sum([sum([1 if i < j and x >= y else 0 for j, y in enumerate(b)]) for i, x in enumerate(b)])
    
    # total number of pairs
    NN = len(a) * (len(a) - 1) / 2
    
    # return the ratio
    return 1. * res / NN

def compute_rank_losses(data, key):
    """ Compute rank losses for a dict with data, referenced to key """
    return {keyother: rank_loss(data[key], data[keyother]) for keyother in data.keys() if keyother != key}

def assert_equal(x, y, name_x = "x", name_y = "y"):
  """ Assert that x == y and if not, pretty-print the error """
  assert x == y, "%s = %s must be equal to %s = %s" % (str(name_x), str(x), str(name_y), str(y))

def add_methods_from(*modules):
    """ Register all methods from modules
        @see http://www.qtrac.eu/pyclassmulti.html
    """
    def decorator(Class):
        for module in modules:
            for method in getattr(module, "__methods__"):
                if hasattr(Class, method.__name__):
                  print(method.__name__)
                  raise Warning("Shadowing a previous method %s by loading module %s" % (str(method.__name__), str(module)))
                setattr(Class, method.__name__, method)
        return Class
    return decorator

def register_method(methods):
    """ Register a method in a class by add_methods_from
        @see http://www.qtrac.eu/pyclassmulti.html
    """
    def register_method(method):
        methods.append(method)
        return method # Unchanged
    return register_method
