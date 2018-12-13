import numpy as np
input_data = np.array([3,5])
weights = {'node_0_0' : np.array([2,-2]),
           'node_0_1' : np.array([2,-1]),
           'node_1_0' : np.array([2, 6]),
           'node_1_1' : np.array([2,9]),
           'output':np.array([3,6])}


def relu(input):
    output1 = max(input,0)

    # Return the value just calculated
    return (output1)
def pls_die(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)
    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    model_output = (hidden_1_outputs * weights['output']).sum()

    # Return model_output
    return (model_output)
output = pls_die(input_data)
print(output)
