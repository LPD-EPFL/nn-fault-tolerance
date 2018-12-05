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

    # test that norm_error(infty) == mean_v1
    assert np.allclose(exp.get_mean_std_error()[0], exp.get_norm_error(ord = np.infty)), "One of implementations of bound v1 (mean) is incorrect"

    # test that |WL|*...*|W_2|*C*p == mean_v2
    R = np.eye(exp.N[-1])
    for w in exp.W[1:][1::-1]:
        R = R @ np.abs(w.T)
    v21 = exp.get_mean_error_v2()
    inp = np.ones(exp.N[1]) if activation == 'sigmoid' else exp.C[0]
    v22 = R @ inp * max(exp.P)
    assert np.allclose(v21, v22), "One of implementations of bound v2 (mean) is incorrect got %s %s" % (str(v21), str(v22))

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


print("All done")
