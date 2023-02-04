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

            self.outputs[node] = total + self.bias[2][node]

        return 0

    def backpropagate(self, outputs):

        # Check for correct size
        if not len(outputs) == len(self.outputs):
            return 1

        # Get errors in predicted values
        errors = np.zeros(len(self.outputs))
        for num in range(0, len(outputs)):
            errors[num] = outputs[num] - self.outputs[num]

        # Array to hold weight changes
        changes = []
        changes.append(np.zeros((len(self.inputs), len(self.hiddenLayer[0]))))
        changes.append(np.zeros((len(self.hiddenLayer[0]), len(self.hiddenLayer[0]))))
        changes.append(np.zeros((len(self.hiddenLayer[0]), len(self.outputs))))

        # Update last set of weights
        for node in range(0, len(self.hiddenLayer[1])):
            for oNode in range(0, len(self.outputs)):
                changes[2][node][oNode] += 2 * self.hiddenLayer[1][node] * (outputs[oNode] - self.outputs[oNode]) * 0.01 

        # Update last set of bias
        for num in range(0, len(self.bias[2])):
            self.bias[2][num] += 2 * (outputs[num] - self.outputs[num]) * 0.01

        # Update middle set of weights
        for outNode in range(0, len(self.outputs)):
            for node in range(0, len(self.hiddenLayer[0])):
                for oNode in range(0, len(self.hiddenLayer[1])):
                    changes[1][node][oNode] += 2 * self.weights[2][oNode][outNode] * (outputs[outNode] - self.outputs[outNode]) * self.hiddenLayer[0][node] * 0.01 

        # Update weights with changes
        for layer in range(0, len(self.weights)):
            for node in range(0, len(self.weights[layer])):
                for weight in range(0, len(self.weights[layer][node])):
                    self.weights[layer][node][weight] += changes[layer][node][weight]
        
        return 0
        
