# standard imports
import numpy as np
import sys
from functools import partial
import pandas as pd
import pickle

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

def accuracy(ys, ys_true):
  """ Get accuracy for array of ys and correct ys """
  assert len(ys.shape) == 1, "Must have vector input (ys)"
  assert len(ys_true.shape) == 1, "Must have vector input (ys_true)"
  zero_one = [y == y_true for y, y_true in zip(ys, ys_true)]
  return 1. * np.sum(zero_one) / len(zero_one)

def matrix_argmax(X):
  """ Return argmax for a matrix """
  return np.unravel_index(X.argmax(), X.shape)

def argmax_accuracy(ys, ys_true):
  """ Get accuracy for one-hot vectors of shape (inputs, outputs) """
  assert len(ys.shape) == 2, "Must have a matrix as input (ys)"
  assert len(ys_true.shape) == 2, "Must have a matrix as input (ys_true)"
  ys = np.argmax(ys, axis = 1)
  ys_true = np.argmax(ys_true, axis = 1)
  return accuracy(ys, ys_true)

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
                # backward compatibility hack: get_bound_bX -> get_bound_v*
                setattr(Class, btov(method.__name__), method)
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

def btov(s):
    """ backward comp function """
    kw = 'get_bound_b'
    if s.startswith(kw):
        return 'get_bound_v' + s[len(kw):]
    return s

def cache_graph(self):
    """ Cache the result of a function in the class, subsequent call to a function will return a cached value """
    caller_name = sys._getframe(1).f_code.co_name

    def memoize_(f):
      # if already have the attribute, return a function which returns it
      def try_from_cache(*args, **kwargs):
        attr = '__cache_' + caller_name + '_' + f.__name__ + '_args_%s_kwargs_%s' % (str(args), str(kwargs))
        if not hasattr(self, attr):
          setattr(self, attr, f(*args, **kwargs))
        attr = '__cache_' + btov(caller_name) + '_' + f.__name__ + '_args_%s_kwargs_%s' % (str(args), str(kwargs))
        if not hasattr(self, attr):
          setattr(self, attr, f(*args, **kwargs))
          #print('Storing %s' % attr)
        return getattr(self, attr)
      return try_from_cache
    return memoize_

def print_shape(r, name):
  """ Print shapes of each element in a dictionary r """
  print('=== Shapes of %s ===' % str(name))
  print(pd.DataFrame([[key, np.array(value).shape] for key, value in r.items()], columns = ['name', 'shape']))

def pickle_w(var, filename):
    """ Write pickle to file, filename w/o extension, using current dir """
    pickle.dump(var, open("%s.pkl" % filename, "wb"))

def pickle_r(filename):
    """ Read from pickle file, filename w/o extension, using current dir """
    return pickle.load(open("%s.pkl" % filename, "rb"))
