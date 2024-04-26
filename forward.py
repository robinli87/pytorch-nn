#Forward propagation through the network
#First import libraries and check hardware compatibility

import torch
import random
import numpy
if torch.cuda.is_available() == True:
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())

else:
    print("CUDA is not available.")

# get inputs, define network structure

structure = [2, 5, 5, 5, 2]

train_input = []
train_output = []

#let's give the inputs and outputs some random stuff for now
for i in range(0, 10):
    #get 10 datapoints. 
    train_input.append([random.random(), random.random()])
    train_output.append([random.random(), random.random()])
    #each datapoint is a 2D vector; both inputs and outputs are 2D

train_input = torch.tensor(train_input).cuda()
train_output = torch.tensor(train_output).cuda()
batch_size = len(train_input)

#generate weight arrays and bias arrays from the given structure.
number_of_layers = len(structure)
param_length = number_of_layers - 1

weights = [None]*(param_length)
biases = [None]*(param_length)

z = [None]*number_of_layers
a = [None]*number_of_layers

for l in range(0, param_length):
    weights[l] = torch.normal(0, 2, size=(structure[l], structure[l+1])).cuda()
    biases[l] = torch.zeros(structure[l+1]).cuda()


def activate(node):
    result = torch.tanh(node)
    return(result)

total_loss = 0
"""
for i in range(0, batch_size):
    #now set the first layer of nodes to be the inputs
    z[0] = train_input[i]
    a[0] = train_input[i] #between z[0] and a[0] we can insert some kind of normalisation function; z is raw input and a is processed input

    #now we are ready for some multiplication
    for l in range(1, number_of_layers):
        #calculate Wa
        next_z_unbiased = torch.matmul(a[l-1], weights[l-1])
        #now next z = Wa + b
        z[l] = torch.add(biases[l-1], next_z_unbiased)
        #activate this node
        a[l] = activate(z[l])

    #we have reached the end. extract results (last layer) and compare with the expected output
    error_vector = torch.subtract(a[-1], train_output[i])
    total_loss += torch.dot(error_vector, error_vector)

print(total_loss)
 """

def forward_propagation(train_input, weights, biases):
    z = [None]*number_of_layers
    a = [None]*number_of_layers

    
    out = [None]*len(train_input)

    for i in range(0, batch_size):
    #now set the first layer of nodes to be the inputs
    # this is going to be one run-through with one input feature vector
        #between z[0] and a[0] we can insert some kind of normalisation function; z is raw input and a is processed input
        z[0] = train_input[i]
        a[0] = train_input[i]

        #now we are ready for some multiplication
        for l in range(1, number_of_layers):
            #calculate Wa
            next_z_unbiased = torch.matmul(a[l-1], weights[l-1])
            #now next z = Wa + b
            z[l] = torch.add(biases[l-1], next_z_unbiased)
            #activate this node
            a[l] = activate(z[l])

        out[i] = z[-1]

    return(out)

out = forward_propagation(train_input, weights, biases)
print(out)

