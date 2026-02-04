import numpy as np


#[Study,Sleep,Attendance]
inputs = np.array([0.9,0.2,0.8])

#Determine the importance of each input is
weights = np.array([2.5,1.5,0.5])

#Test is difficult. It will be fail until the inputs are high enough
bias = -2.0

weighted_sum = (inputs[0]*weights[0]) + (inputs[1]*weights[1] + inputs[2]*weights[2]) + bias

print(f"Weighted Sum (Z): {weighted_sum}")

activation = 1.0 / (1.0 + np.exp(-weighted_sum))

print(f"Prediction (A): {np.around(activation, 4)}")

np.random.seed(42)

def initialize_network(num_inputs,num_hidden_layers,num_nodes_hidden,num_nodes_outputs):
    num_nodes_previous = num_inputs
    network = {}

    for layer in range(num_hidden_layers+1):

        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_outputs
        else:
            layer_name = 'layer_{}'.format(layer+1)
            num_nodes = num_nodes_hidden[layer]


        network[layer_name] = {}

        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous),decimals=2),
                'bias': np.around(np.random.uniform(size=1),decimals=2)
            }

        num_nodes_previous = num_nodes

    return network

def compute_weighted_sum(inputs,weights,bias):
    return np.sum(inputs*weights) + bias

def node_activation(weighted_sum):
    return 1.0/(1.0+np.exp(-1 * weighted_sum))


def forward_propagate(network, inputs):
    layer_inputs = np.array(inputs)
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print(f'   Output of {layer}: {layer_outputs}')

        layer_inputs = layer_outputs

    return layer_outputs

alice_inputs = np.array([0.9,0.2,0.8])

print(f"--- Predicting for Student (Inputs: {alice_inputs}) ---")

student_network = initialize_network(num_inputs=3,num_hidden_layers=2,num_nodes_hidden=[4,3],num_nodes_outputs=1)


prediction = forward_propagate(network=student_network,inputs=alice_inputs)

print(f"\nFinal Probability: {prediction[0]}")

if prediction[0] > 0.5:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
