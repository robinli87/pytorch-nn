# neural network with pytorch tensors

import torch
import random
import numpy

#check if torch can run:
if torch.cuda.is_available() == True:
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())

else:
    print("CUDA is not available.")
    exit()

#define class. To initialise, we need to know the shape of this network
class NN:
    def __init__(self, structure):
        self.structure = structure
        #now we need to generate weights, nodes, biases according to the structure
        self.number_of_layers = len(structure)
        self.param_length = self.number_of_layers - 1

        self.weights = [None]*(self.param_length)
        self.biases = [None]*(self.param_length)

        for l in range(0, param_length):
            self.weights[l] = torch.normal(0, 2, size=(structure[l], structure[l+1])).cuda()
            self.biases[l] = torch.zeros(structure[l+1]).cuda()

    def forward_propagation(self, this_input, weights, biases):

        z = [None]*self.number_of_layers
        a = [None]*self.number_of_layers
        #out = [None]*batch_size
        #batch_size = len(train_input)

        #now set the first layer of nodes to be the inputs
        # this is going to be one run-through with one input feature vector
        #between z[0] and a[0] we can insert some kind of normalisation function; z is raw input and a is processed input
        z[0] = this_input
        a[0] = this_input

        #now we are ready for some multiplication
        for l in range(1, self.number_of_layers):
            #calculate Wa
            next_z_unbiased = torch.matmul(a[l-1], weights[l-1])
            #now next z = Wa + b
            z[l] = torch.add(biases[l-1], next_z_unbiased)
            #activate this node
            a[l] = self.activate(z[l])


        return(z[-1])

    def activate(self, nodes):
        #the activation function. Let's use tanh for now
        activated = torch.tanh(nodes)
        return(activated)


    def loss(self, train_input, train_output, local_weights, local_biases):
        #run the network once with all of these inputs
        out = forward_propagation(train_input, local_weights, local_biases)
        diff = train_output - out


