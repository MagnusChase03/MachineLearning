import matplotlib.pyplot as plt
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def sigmoid_dir(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, inputNum, hiddenLayerNum, outputNum, lr):
    
        self.learningRate = lr        

        # Create default layers
        self.inputs = np.zeros(inputNum)
        self.hiddenLayer = np.zeros((2, hiddenLayerNum))
        self.outputs = np.zeros(outputNum)

        # Random Weights
        self.weights = []
        self.weights.append(np.random.rand(inputNum, hiddenLayerNum))
        self.weights.append(np.random.rand(hiddenLayerNum, hiddenLayerNum))
        self.weights.append(np.random.rand(hiddenLayerNum, outputNum))

        # Random Bias
        self.bias = []
        self.bias.append(np.zeros(hiddenLayerNum))
        self.bias.append(np.zeros(hiddenLayerNum))
        self.bias.append(np.zeros(outputNum))

        # Matplotlib graph
        self.fig, self.ax = plt.subplots()

    def forward(self, inputs):

        # Make sure input size is correct
        if not len(inputs) == len(self.inputs):
            return 1

        self.inputs = inputs

        # Pass to first hidden layer
        hiddenLayer0 = inputs.dot(self.weights[0])
        hiddenLayer0 += self.bias[0]
        hiddenLayer0 = sigmoid(hiddenLayer0)

        self.hiddenLayer[0] = hiddenLayer0

        # Pass to second hiddenLayer
        hiddenLayer1 = self.hiddenLayer[0].dot(self.weights[1]) 
        hiddenLayer1 += self.bias[1]
        hiddenLayer1 = sigmoid(hiddenLayer1)

        self.hiddenLayer[1] = hiddenLayer1

        # Outputs
        outputs = self.hiddenLayer[1].dot(self.weights[2])
        outputs += self.bias[2]

        self.outputs = outputs

        return 0

    def backprop(self, outputs):
        
        # Make size check
        if not len(outputs) == len(self.outputs):
            return 1

        # Get difference
        errors = outputs - self.outputs

        # Update last layer and bias
        changes = []
        for error in range(0, len(errors)):
            changes.append(errors[error] * self.hiddenLayer[1])
            self.bias[2] += errors[error] * self.learningRate

        changes = np.array(changes)
        changes = changes.T * self.learningRate
        self.weights[2] += changes

        return 0
