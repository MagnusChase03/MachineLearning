import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

class NeuralNetwork:
    def __init__(self, inputNum, hiddenLayerNum, outputNum):
        
        # Create default layers
        self.inputs = np.zeros(inputNum)
        self.hiddenLayer = np.zeros((2, hiddenLayerNum))
        self.outputs = np.zeros(outputNum)

        # Random Weights
        self.weights = []
        self.weights.append(np.random.rand(inputNum, hiddenLayerNum))
        self.weights.append(np.random.rand(hiddenLayerNum, hiddenLayerNum))
        self.weights.append(np.random.rand(hiddenLayerNum, hiddenLayerNum))
        self.weights.append(np.random.rand(hiddenLayerNum, outputNum))

        # Random Bias
        self.bias = []
        self.bias.append(np.random.rand(hiddenLayerNum))
        self.bias.append(np.random.rand(hiddenLayerNum))
        self.bias.append(np.random.rand(outputNum))

    def forward(self, inputs):

        # Make sure input size is correct
        if not len(inputs) == len(self.inputs):
            return 1

        self.inputs = inputs

        # Pass to first hidden layer
        for node in range(0, len(self.hiddenLayer[0])):
            total = 0.0
            for iNode in range(0, len(self.inputs)):
                total += self.inputs[iNode] * self.weights[0][iNode][node]

            self.hiddenLayer[0][node] = sigmoid(total + self.bias[0][node])

        # Pass to second hidden layer
        for node in range(0, len(self.hiddenLayer[1])):
            total = 0.0
            for iNode in range(0, len(self.hiddenLayer[0])):
                total += self.hiddenLayer[0][iNode] * self.weights[1][iNode][node]

            self.hiddenLayer[1][node] = sigmoid(total + self.bias[1][node])

        # Pass to output layer
        for node in range(0, len(self.outputs)):
            total = 0.0
            for iNode in range(0, len(self.hiddenLayer[1])):
                total += self.hiddenLayer[1][iNode] * self.weights[2][iNode][node]

            self.outputs[node] = sigmoid(total + self.bias[2][node])

        return 0
        
