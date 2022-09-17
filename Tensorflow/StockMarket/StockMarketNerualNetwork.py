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

    # Activation Function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Predict next day high given last two day
    def predict(self, inputs):
        
        # Handle first layer
        total = 0
        layer = []
        for i in range(0, 20):

            total += inputs[(i % 10)] * self.weights[0][i]

            if i == 9 or i == 19:
                total += self.bias[i % 9]
                layer.append(self.sigmoid(total))
                total = 0

        # Handle second layer
        return (layer[0] * self.weights[1][0]) + (layer[1] * self.weights[1][1]) 