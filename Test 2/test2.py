import numpy as np

### NO HIDDEN LAYER EXAMPLE

# sigmoid function
def sigmoid(x, deriv=False):
	if deriv == True:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

# input dataset
x = np.array([	[0, 0, 1],
				[0, 1, 1],
				[1, 0, 1],
				[1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# this time we want weights with value in [-1, 1]
np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1 # syn0 = synapse 0


for iter in xrange(100000):
	# forward propagation
	l0 = x;
	l1 = sigmoid(np.dot(l0, syn0))

	# error calculation
	l1_error = y - l1

	# evaluate the slope
	l1_delta = l1_error * sigmoid(l1, True)

	# update weights
	syn0 += np.dot(l0.T, l1_delta)


print "Output after training : \n" + str(l1)
