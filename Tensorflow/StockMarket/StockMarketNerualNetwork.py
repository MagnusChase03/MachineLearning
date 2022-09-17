import random
import math

class StockMarketNerualNetwork:

    def __init__(self):

        # Create random weights
        self.weights = [[], []]
        for i in range(0, 20):
            self.weights[0].append(random.random())

        for i in range(0, 2):
            self.weights[1].append(random.random())

        # Create random bias
        self.bias = []
        for i in range(0, 2):
            self.bias.append(random.random())

        self.layer = [0.0, 0.0]

    # Activation Function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Predict next day high given last two day
    def predict(self, inputs):
        
        # Handle first layer
        total = 0
        for i in range(0, 20):

            total += inputs[(i % 10)] * self.weights[0][i]

            if i == 9 or i == 19:
                total += self.bias[i % 9]
                self.layer[i % 9] = self.sigmoid(total)
                total = 0

        # Handle second layer
        return (self.layer[0] * self.weights[1][0]) + (self.layer[1] * self.weights[1][1]) 

    # Train network using backpropagation
    def train(self, inputs, outputs):
        
        for i in range(0, len(inputs)):

            prediction = self.predict(inputs[i])
            error = outputs[i] - prediction

            # Fix second set of weights and bias
            self.weights[1][0] += 0.01 * error * self.layer[0]
            self.weights[1][1] += 0.01 * error * self.layer[1]
            self.bias[0] += 0.01 * error
            self.bias[1] += 0.01 * error

            # Fix first set of weights
            for j in range(0, 20):
                self.weights[0][j] += 0.01 * error * inputs[i][j % 10]