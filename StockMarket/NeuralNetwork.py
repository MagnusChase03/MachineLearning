import matplotlib.pyplot as plt
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

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
        self.bias.append(np.random.rand(hiddenLayerNum))
        self.bias.append(np.random.rand(hiddenLayerNum))
        self.bias.append(np.random.rand(outputNum))

        # Matplotlib graph
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.iteration = 0

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
            errors[num] = np.power(outputs[num] - self.outputs[num], 2)
        
        # Add error points to graph
        if self.iteration % 100 == 0:
            for node in range(0, len(self.outputs)):
                self.ax.plot([self.iteration], [errors[node]], 'go')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # Array to hold weight changes
        changes = []
        changes.append(np.zeros((len(self.inputs), len(self.hiddenLayer[0]))))
        changes.append(np.zeros((len(self.hiddenLayer[0]), len(self.hiddenLayer[0]))))
        changes.append(np.zeros((len(self.hiddenLayer[0]), len(self.outputs))))

        changes.append(np.zeros(len(self.hiddenLayer[0])))
        changes.append(np.zeros(len(self.hiddenLayer[0])))
        changes.append(np.zeros(len(self.outputs)))

        # Update last set of weights
        for node in range(0, len(self.hiddenLayer[1])):
            for oNode in range(0, len(self.outputs)):
                changes[2][node][oNode] += 2 * self.hiddenLayer[1][node] * (outputs[oNode] - self.outputs[oNode]) * self.learningRate 

        # Update last set of bias
        for num in range(0, len(self.bias[2])):
            #self.bias[2][num] += 2 * (outputs[num] - self.outputs[num]) * self.learningRate
            changes[5][num] += 2 * (outputs[num] - self.outputs[num]) * self.learningRate

        # Update middle set of weights
        for outNode in range(0, len(self.outputs)):
            for node in range(0, len(self.hiddenLayer[0])):
                for oNode in range(0, len(self.hiddenLayer[1])):
                    changes[1][node][oNode] += 2 * self.weights[2][oNode][outNode] * (outputs[outNode] - self.outputs[outNode]) * self.hiddenLayer[0][node] * self.learningRate

        # Update middle bias
        for outNode in range(0, len(self.outputs)):
            for oNode in range(0, len(self.hiddenLayer[1])):
                #self.bias[1] += 2 * self.weights[2][oNode][outNode] * (outputs[outNode] - self.outputs[outNode]) * self.learningRate
                changes[4][oNode] += 2 * self.weights[2][oNode][outNode] * (outputs[outNode] - self.outputs[outNode]) * self.learningRate

        # Update first bias
        for outNode in range(0, len(self.outputs)):
            for oNode in range(0, len(self.hiddenLayer[1])):
                for node in range(0, len(self.hiddenLayer[0])):
                    #self.bias[0] += 2 * self.weights[2][oNode][outNode] * (outputs[outNode] - self.outputs[outNode]) * self.weights[1][node][oNode] * self.learningRate
                    changes[3][node] += 2 * self.weights[2][oNode][outNode] * (outputs[outNode] - self.outputs[outNode]) * self.weights[1][node][oNode] * self.learningRate
        
        return changes

    def update(self, changes):

        # Update weights with changes
        for change in range(0, len(changes)):
            for layer in range(0, len(self.weights)):
                for node in range(0, len(self.weights[layer])):
                    for weight in range(0, len(self.weights[layer][node])):
                        self.weights[layer][node][weight] += changes[change][layer][node][weight]
            for node in range(0, len(changes[change][5])):
                self.bias[2][node] += changes[change][5][node]

            for node in range(0, len(changes[change][4])):
                self.bias[1][node] += changes[change][4][node]

            for node in range(0, len(changes[change][3])):
                self.bias[0][node] += changes[change][3][node]

    def train(self, inputs, outputs, rounds, batchSize):

        # Make sure datasets are correct size
        if not len(inputs) == len(outputs) or len(inputs) == 0:
            return 1

        self.interation = 0

        # Train model
        for i in range(0, rounds):
            
            batchChanges = []
            for dataset in range(0, len(inputs)):
                if self.forward(inputs[dataset]) == 1:
                    return 1

                result = self.backpropagate(outputs[dataset])
                if result == 1:
                    return 1
                else:
                    batchChanges.append(result)
    
                if len(batchChanges) == batchSize or dataset == len(inputs) - 1:
                    self.update(batchChanges)
                    batchChanges = []

            self.iteration += 1

        plt.ioff()
        plt.show()

        return 0
        
