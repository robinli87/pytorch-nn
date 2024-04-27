# neural network with pytorch tensors

import torch
import random
import numpy
import time

# mp.set_start_method('spawn')
import torch.multiprocessing as mp

# check if torch can run:
if torch.cuda.is_available() == True:
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())

else:
    print("CUDA is not available.")
# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define class. To initialise, we need to know the shape of this network


class NN:
    def __init__(self, structure):
        self.structure = structure
        # now we need to generate weights, nodes, biases according to the structure
        self.number_of_layers = len(structure)
        self.param_length = self.number_of_layers - 1

    def load_parameters(self, option="generate"):

        self.weights = [None]*(self.param_length)
        self.biases = [None]*(self.param_length)

        for l in range(0, self.param_length):
            self.weights[l] = torch.normal(
                0, 2, size=(self.structure[l], self.structure[l+1]), device=device)
            self.biases[l] = torch.zeros(self.structure[l+1], device=device)

        return (self.weights, self.biases)

    def forward_propagation(self, this_input, weights, biases):

        z = [None]*self.number_of_layers
        a = [None]*self.number_of_layers
        # out = [None]*batch_size
        # batch_size = len(train_input)

        # now set the first layer of nodes to be the inputs
        # this is going to be one run-through with one input feature vector
        # between z[0] and a[0] we can insert some kind of normalisation function; z is raw input and a is processed input
        z[0] = this_input
        a[0] = this_input

        # now we are ready for some multiplication
        for l in range(1, self.number_of_layers):
            # calculate Wa
            next_z_unbiased = torch.matmul(a[l-1], weights[l-1])
            # now next z = Wa + b
            z[l] = torch.add(biases[l-1], next_z_unbiased)
            # activate this node
            a[l] = self.activate(z[l])

        return (z[-1])

    def activate(self, nodes):
        # the activation function. Let's use tanh for now
        activated = torch.tanh(nodes)
        return (activated)

    def loss(self, train_input, train_output, local_weights, local_biases):
        # run the network once with all of these inputs
        self.batch_size = len(train_input)
        # train_input = torch.tensor(train_input).cuda()
        # train_output = torch.tensor(train_output).cuda()
        total = 0
        for i in range(0, self.batch_size):
            out = self.forward_propagation(
                train_input[i], local_weights, local_biases)
            diff = torch.subtract(train_output[i], out)
            error = torch.dot(diff, diff)
            total += float(error)

        return (total)

    def gradient_calculation(self, l, k):
        # gradients = torch.zeros(self.structure[l+1]).cuda()
        for j in range(0, self.structure[l]):
            # calculate gradient
            w1 = self.weights.copy()
            w1[l][j][k] += self.dw
            L1 = self.loss(self.train_input,
                           self.train_output, w1, self.biases)
            w1[l][j][k] += -2 * self.dw
            L2 = self.loss(self.train_input,
                           self.train_output, w1, self.biases)
            gradient = (L1 - L2) / (2 * self.dw)
            self.weights[l][j][k] += -self.learning_rate * gradient

        b1 = self.biases.copy()
        b1[l][k] += self.db
        B1 = self.loss(self.train_input,
                       self.train_output, self.weights, b1)
        b1[l][k] += -2 * self.db
        B2 = self.loss(self.train_input,
                       self.train_output, self.weights, b1)
        gradient = (B1 - B2) / (2 * self.db)

        self.biases[l][k] += -self.learning_rate * gradient

    def backpropagation(self):

        processes = []

        for l in range(0, self.param_length):
            for k in range(0, self.structure[l+1]):
                # p = mp.Process(target=self.gradient_calculation, args=(l, k))
                self.gradient_calculation(l, k)
                #         p.start()
                #         processes.append(p)
                #
                # for p in processes:
                #     p.join()

    def train(self, train_input, train_output):
        # firstly, share arrays in memory:
        for item in self.weights:
            item.share_memory_()

        for item in self.biases:
            item.share_memory_()

        # run once for a benchmark
        self.train_input = train_input
        self.train_output = train_output
        benchmark = self.loss(train_input, train_output,
                              self.weights, self.biases)

        epoch = 0
        self.learning_rate = 0.00012
        self.dw = 0.000001
        self.db = 0.000001

        # now carry out backpropagation once
        self.backpropagation()
        new_loss = self.loss(train_input, train_output,
                             self.weights, self.biases)

        print("bench ", benchmark)
        print("new ", new_loss)

        # loop until we have reached the tolerance level required:
        while new_loss < benchmark:
            benchmark = new_loss
            start = time.time()
            self.backpropagation()
            print("Time consumed: ", time.time()-start)
            new_loss = self.loss(train_input, train_output,
                                 self.weights, self.biases)
            epoch += 1
            print(new_loss)


structure = [10, 20, 100, 100, 10]
# train_input = []
# train_output = []

# # let's give the inputs and outputs some random stuff for now
# for i in range(0, 10):
#     # get 10 datapoints.
#     train_input.append([random.random(), random.random()])
#     train_output.append([random.random(), random.random()])
#     # each datapoint is a 2D vector; both inputs and outputs are 2D
#
# train_input = torch.tensor(train_input, device=device)
# train_output = torch.tensor(train_output, device=device)

train_input = torch.normal(0, 1, size=(10, structure[0])).cuda()
train_output = torch.normal(0, 1, size=(10, structure[-1])).cuda()
# print(train_input.size())

AI = NN(structure)
weights, biases = AI.load_parameters()
AI.forward_propagation(train_input, weights, biases)
loss = AI.loss(train_input, train_output, weights, biases)
AI.train(train_input, train_output)
