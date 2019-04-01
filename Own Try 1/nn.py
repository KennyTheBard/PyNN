import numpy as np

class NeuralNetwork(object):
	def __init__(self, inputSize, hiddenSize, outputSize):
		self.inputSize = inputSize
		self.hiddenSize = hiddenSize
		self.outputSize = outputSize

		self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
		self.w2 = np.random.randn(self.hiddenSize, self.outputSize)

	def sigmoid(self, s):
		return 1 / (1 + np.exp(-s))

	def sigmoidPrime(self, s):
		return s / (1 - s)

	def forward(self, x):
		self.z1 = self.sigmoid(np.dot(x, self.w1))
		self.z2 = self.sigmoid(np.dot(self.z1, self.w2))
		return self.z2

	def backward(self, x, y, out):
		self.out_delta = (y - out) * self.sigmoidPrime(out)
		self.z2_delta = np.dot(self.out_delta, self.w2.T) * self.sigmoidPrime(self.z2)

		self.w1 += x.T.dot(self.z2_delta)
		self.w2 += self.z2.T.dot(self.out_delta)

	def train(self, x, y):
		out = self.forward(x)
		self.backward(x, y, out)

	def predict(self, x):
		print str(self.forward(x))


def f(x, y):
	return np.exp(1 - x * x - y * y)

list_x = []
list_y = []
for i in range(20):
	x = np.random.uniform(-3, 3)
	y = np.random.uniform(-3, 3)
	list_x.append([x, y])
	list_y.append([f(x, y)])

x = np.asarray(list_x, dtype = float)
x = x / np.amax(x, axis = 0)

y = np.asarray(list_y, dtype = float)
y = y / np.amax(y, axis = 0)

NN = NeuralNetwork(2, 7, 1)

for u in xrange(10000):
	NN.train(x, y)
	if u % 100 == 0:
		pred = np.array(([1, 1]), dtype = float)
		pred = pred / np.amax(pred, axis = 0)
		NN.predict(pred)
		print "Correct: " + str(f(1, 1))
