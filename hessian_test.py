# get a tf session
sess = get_session()

# input vector
x = tf.Variable(np.random.randn(5, 1))

# symmetrical matrix
A = np.random.randn(5, 5)
A = (A + A.T) / 2

# out=x.TAx
y = (tf.transpose(x) @ A @ x)

# computing gradients
sess.run(tf.global_variables_initializer())
sess.run(tf.gradients(y, x))

# computing the hessian
H = sess.run(tf.hessians(y, x))[0].reshape(5, 5)

# sanity check
assert np.allclose(A.T * 2, H), "Hessian is invalid"
