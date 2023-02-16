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
        errors.resize(1, len(errors))

        # Update last layer and bias
        hiddenLayer1 = np.copy(self.hiddenLayer[1])
        hiddenLayer1.resize(1, len(hiddenLayer1))

        changes = hiddenLayer1.T.dot(errors)
        self.weights[2] += changes * self.learningRate

        for error in range(0, len(errors)):
            self.bias[2] += errors[error] * self.learningRate

        # Get Layer 1 Error
        hiddenLayer1Errors = self.weights[2].dot(errors.T)
        hiddenLayer1Errors = hiddenLayer1Errors.T 

        # Update middle layer and bias
        hiddenLayer0 = np.copy(self.hiddenLayer[0])
        hiddenLayer0.resize(1, len(hiddenLayer0))

        changes = hiddenLayer0.T.dot(hiddenLayer1Errors)
        self.weights[1] += changes * self.learningRate

        for error in range(0, len(hiddenLayer1Errors)):
            self.bias[1] += hiddenLayer1Errors[error] * self.learningRate

        # Hidden layer 0 erros
        hiddenLayer0Errors = self.weights[1].dot(hiddenLayer1Errors.T)
        hiddenLayer0Errors = hiddenLayer0Errors.T

        # Update first layer and bais
        inp = np.copy(self.inputs)
        inp.resize(1, len(inp))

        changes = inp.T.dot(hiddenLayer0Errors)
        self.weights[0] += changes * self.learningRate

        for error in range(0, len(hiddenLayer0Errors)):
            self.bias[0] += hiddenLayer0Errors[error] * self.learningRate

        return 0
