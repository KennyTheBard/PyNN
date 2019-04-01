import numpy as np

# x = (hours sleeped, hours studied)
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
# y = score on test
y = np.array(([92], [86], [89]), dtype=float)

# scale units
x = x / np.amax(x, axis=0) # maximum of x array
y = y / 100;

class NeuralNetwork(object):
	def __init__(self):
		# parameters
		self.inputSize = 2
		self.hiddenSize = 3
		self.outputSize = 1

		# weights
		self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
		self.w2 = np.random.randn(self.hiddenSize, self.outputSize)


	def sigmoid(self, s):
		return 1 / (1 + np.exp(-s))


	def forward(self, x):
		# forward propagation
		self.z = np.dot(x, self.w1) # input to hidden
		self.z2 = self.sigmoid(self.z) # activation function
		self.z3 = np.dot(self.z2, self.w2) # hidden to output
		out = self.sigmoid(self.z3) # activation function
		return out


	def sigmoidPrime(self, s):
		# derivative of sigmoid => slope
		return s * (1 - s)


	def backward(self, x, y, out):
		# backward propagation
		self.o_error = y - out # error in output
		self.o_delta = self.o_error * self.sigmoidPrime(out)

		self.z2_error = self.o_delta.dot(self.w2.T) # error in hidden layer
		self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

		self.w1 += x.T.dot(self.z2_delta) # adjust input to hidden weights
		self.w2 += self.z2.T.dot(self.o_delta) # adjust hidden to output weights


	def train(self, x, y):
		out = self.forward(x)
		self.backward(x, y, out)


	def predict(self, x):
		print "Input  : " + str(x)
		print "Output : " + str(self.forward(x))


	def saveWeights(self):
		np.savetxt("w1.txt", self.w1, fmt="%s")
		np.savetxt("w2.txt", self.w2, fmt="%s")



NN = NeuralNetwork()
for u in xrange(1000):
	print "Output: " + str(NN.forward(x))
	print "Loss: " + str(np.mean(np.square(y - NN.forward(x))))
	NN.train(x, y)

xPred = np.array(([1,5]), dtype=float)
xPred = xPred / np.amax(xPred, axis=0)
NN.predict(xPred)

print x
print xPred
