import numpy as np
import math

class NeuralNetwork(object):
	def __init__(self, inputSize, hiddenLayers, outputSize):
		self.layers = [inputSize] + hiddenLayers + [outputSize]
		print self.layers
		self.weights = [0] * (len(self.layers) - 1)
		for i in range(len(self.weights)):
			self.weights[i] = np.random.randn(self.layers[i], self.layers[i + 1])
		self.zs = [0] * len(self.weights)
		self.deltas = [0] * len(self.weights)

	def sigmoid(self, s):
		return 1 / (1 + np.exp(-s))

	def sigmoidPrime(self, s):
		return s * (1 - s)

	def forward(self, x):
		for i in range(len(self.weights)):
			if i == 0:
				self.zs[i] = self.sigmoid(np.dot(x, self.weights[i]))
			else :
				self.zs[i] = self.sigmoid(np.dot(self.zs[i - 1], self.weights[i]))
		return self.zs[-1]

	def backward(self, x, y, out):
		for i in reversed(range(len(self.weights))):
			if i == len(self.weights) - 1:
				self.deltas[i] = (y - out) * self.sigmoidPrime(out)
			else :
				self.deltas[i] = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
				self.deltas[i] *= self.sigmoidPrime(self.zs[i])

		for i in range(len(self.weights)):
			if i == 0:
				self.weights[i] += x.T.dot(self.deltas[i])
			else:
				self.weights[i] += self.zs[i].T.dot(self.deltas[i])


	def train(self, x, y):
		out = self.forward(x)
		self.backward(x, y, out)

	def predict(self, x):
		print str(self.forward(x))


def f(x, y, z):
	return 5 * x - 2 * y + z

list_x = []
list_y = []
for i in range(20):
	x = np.random.uniform(1, 10)
	y = np.random.uniform(1, 10)
	z = np.random.uniform(1, 10)
	list_x.append([x, y, z])
	list_y.append([f(x, y, z)])

x = np.asarray(list_x, dtype = float)
x = x / np.amax(x, axis = 0)

y = np.asarray(list_y, dtype = float)
y = y / np.amax(y, axis = 0)

NN = NeuralNetwork(3, [7], 1)

for u in xrange(100):
	NN.train(x[u % len(x)], y[u % len(y)])
	if u % 10 == 0:
		NN.predict(x[0])
		print "Correct: " + str(y[0]) + "\n"
