import numpy as np

np.random.seed(42)

weights = np.around(np.random.uniform(size=6),decimals=2)

biases = np.around(np.random.uniform(size=3),decimals=2)

print(weights)

print(biases)

x1 = 0.5

x2 = 0.85

print('x1 is {} and x2 in {}'.format(x1,x2))

z11 = x1 * weights[0] + x2 * weights[1] + biases[0]

print('The weighted sum of the inputs inputs at the first node is {}'.format(z11))

z12 = x1 * weights[2] + x2 * weights[3] + biases[1]

print('The weighted sum of the inputs inputs at the second node is {}'.format(z12))

a11 = 1.0 / (1.0 + np.exp(-z11))

print('The activation at the first node is {}'.format(np.around(a11,decimals=2)))

a12 = 1.0 / (1.0 + np.exp(-z12))

print('The activation at the second node is {}'.format(np.around(a12,decimals=2)))

z2 = a11 * weights[4] + a12 * weights[5] + biases[2]

print('The weighted sum at the inputs at the node in the output layer is {}'.format(np.around(z2,decimals=2)))

a2 = 1.0 / (1.0 + np.exp(-z2))

print('The output of the network for x1 = 0.5  and x2 = 0.85 is {}'.format(np.around(a2,decimals=2)))

#Building a Neural Network

n = 2 # number of inputs

num_hiddenlayers = 2 #num of hidden layers

m = [2,2] #num of nodes in each hidden layer


