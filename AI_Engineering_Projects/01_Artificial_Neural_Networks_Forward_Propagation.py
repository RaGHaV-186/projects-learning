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
#
# n = 2 # number of inputs
#
# num_hiddenlayers = 2 #num of hidden layers
#
# m = [2,2] #num of nodes in each hidden layer
#
# num_nodes_output = 1
#
# num_nodes_previous = n
#
# network = {}
#
# for layer in range(num_hiddenlayers + 1):
#
#     if layer == num_hiddenlayers:
#         layer_name = 'output'
#         num_nodes = num_nodes_output
#     else:
#         layer_name = 'layer_{}'.format(layer+1)
#         num_nodes = m[layer]
#
#     network[layer_name] = {}
#
#     for node in range(num_nodes):
#         node_name = 'node_{}'.format(node+1)
#         network[layer_name][node_name] = {
#             'weights':np.around(np.random.uniform(size=num_nodes_previous),decimals=2),
#             'bias':np.around(np.random.uniform(size=1),decimals=2)
#         }
#
#     num_nodes_previous = num_nodes
#
# print(network)


def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs  # number of nodes in the previous layer

    network = {}

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = 'output'  # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)  # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer]

            # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network  # return the network


small_network = initialize_network(5,3,[3,2,3],1)

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

inputs = np.around(np.random.uniform(size=5), decimals=2)

print('The inputs to the network are {}'.format(inputs))

node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']

weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))
