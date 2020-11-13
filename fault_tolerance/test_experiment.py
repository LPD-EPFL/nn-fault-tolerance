from experiment_random import *

# some pfail
p = 1e-2

# loop over activation fcns
for activation in ['sigmoid', 'relu']:
 for _ in range(3):
    # creating an experiment...
    exp = RandomExperiment([50, 40, 30, 5], [p, 0], 1, activation = activation)
    if activation == 'relu': exp.update_C(exp.get_inputs(1000))

    # and some input
    x0 = np.random.randn(exp.N[0], 1) * 5
    x0f = x0.flatten()

    # checking that forward pass is implemented properly
    assert np.linalg.norm(exp.forward_pass_manual(x0).reshape(-1) - exp.predict_no_dropout(x0.reshape(-1))) < 1e-5, "Forward pass does not work properly: manual and TF values disagree"

    # TEST exact expectation: tf implementation, O(p^2) implementation
    errors = exp.get_error(x0.reshape(-1), repetitions = 10000)
    exp_mean = np.mean(errors, axis = 0)

    # "better" bound, O(N) time
    res_better = exp.get_exact_error_v3_better(x0f)

    # tensorflow implementation of orig
    res_tf = exp.get_exact_error_v3_tf(x0f)

    # orig "exact" bound
    res_orig = exp.get_exact_error_v3(x0f).reshape(1, -1)

    # checking that they are close enough...
    tol = 0.5
    assert np.mean(np.abs(res_better - exp_mean)) / np.max(np.abs(exp_mean)) < tol, "Attempt %d: failed to test O(p^2) solution" % _
    assert np.mean(np.abs(res_tf - exp_mean)) / np.max(np.abs(exp_mean)) < tol, "Attempt %d: failed to test TF solution" % _
    assert np.mean(np.abs(res_orig - exp_mean)) / np.max(np.abs(exp_mean)) < tol, "Attempt %d: too far from the mean" % _

    # TEST exact std: tf, O(p^2) and W^2
    exp_std = np.std(errors, axis = 0)
    res_better = exp.get_exact_std_error_v3_better(x0f)
    res_tf = exp.get_exact_std_error_v3_tf(x0f)

    # checking that they are close enough...
    tol = 0.5
    assert np.mean(np.abs(res_better - exp_std)) / np.max(np.abs(exp_std)) < tol, "Attempt %d: failed to test O(p^2) solution" % _
    assert np.mean(np.abs(res_tf - exp_std)) / np.max(np.abs(exp_std)) < tol, "Attempt %d: failed to test TF solution" % _

# testing generate_params
inp = {'a': [0, 1, 2], 'b': ['x', 'y'], 'c': [None]}
out = list(generate_params(**inp))
true_out = [{'c': None, 'b': 'x', 'a': 0},
 {'c': None, 'b': 'y', 'a': 0},
 {'c': None, 'b': 'x', 'a': 1},
 {'c': None, 'b': 'y', 'a': 1},
 {'c': None, 'b': 'x', 'a': 2},
 {'c': None, 'b': 'y', 'a': 2}]
assert out == true_out, "Generate Params must work"

# rank loss tests
assert rank_loss([1,2,3],[3,2,1]) == 1
assert rank_loss([1,2,3],[1,2,3]) == 0
assert rank_loss([1,2,3],[1,2,3]) == 0
assert rank_loss([1,2,3],[2,1,3]) == 1. / 3
assert rank_loss([1,2,3],[2,2,2]) == 1
np.random.seed(0)
assert np.abs(rank_loss(np.random.randn(1000), np.random.randn(1000)) - 0.5) < 0.1

# example class
class A():
  def __init__(self, x = 1):
    self.x = x
  def calc_y(self):
   @cache_graph(self)
   def helper():
     print("Running helper...")
     return self.x * self.x
   return helper()

# checking cache_graph
exp = A(1)
exp1 = A(2)

assert exp.calc_y() == 1
assert exp1.calc_y() == 4
assert exp.calc_y() == 1
assert exp1.calc_y() == 4

# accuracy tests
assert accuracy(np.array([1,2,3]),np.array([4,5,3])) == 1. / 3
assert accuracy(np.array([1,2,3]),np.array([4,5,6])) == 0.
assert argmax_accuracy(np.array([[0,0,1], [1, 0, 0], [0, 1, 0]]), np.array([[0,2,1], [1, 0, 0], [0, 1, 0]])) == 2. / 3

print("All done")
