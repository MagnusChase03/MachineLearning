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

    # Derivative of activation Function
    def sigmoidDer(self, x):
        return x * (1 - x)

    # Predict next day high given last two day
    def predict(self, inputs):

        # print(self.weights)
        
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

            # # Fix second set of weights and bias
            # self.weights[1][0] += 0.01 * error * self.layer[0]
            # self.weights[1][1] += 0.01 * error * self.layer[1]
            # self.bias[0] += 0.01 * error
            # self.bias[1] += 0.01 * error

            # # Fix first set of weights
            # for j in range(0, 20):
            #     self.weights[0][j] += 0.01 * error * inputs[i][j % 10]

            # Using partial derivatives to change weights and bias

            # ∂E/W1 = ∂E/∂Predicted * ∂Predicted/∂s * ∂s/W1
            # ∂E/W2 = ∂E/∂Predicted * ∂Predicted/∂s * ∂s/W2

            # Partial derivatives of squared error (desired - prediction)^2 / 2 where desired is considered a constant
            partialError = prediction - outputs[i]

            # # TODO save this data when predicting to prevent doing again
            layerBeforeSigmoid = [0.0, 0.0]
            total = 0
            for j in range(0, 20):

                total += inputs[i][(j % 10)] * self.weights[0][j]

                if j == 9 or j == 19:
                    total += self.bias[j % 9]
                    layerBeforeSigmoid[j % 9] = total
                    total = 0

            # Derivative of sigmoid
            partialPrediction = self.sigmoidDer(layerBeforeSigmoid[0])
            partialPrediction2 = self.sigmoidDer(layerBeforeSigmoid[1])

            # Derivative of ∂s/W1 is the input given for specifed weight
            # Fixing second set of weights
            self.weights[1][0] += 0.01 * error * self.layer[0]
            self.weights[1][1] += 0.01 * error * self.layer[1]
            self.bias[0] += 0.01 * error
            self.bias[1] += 0.01 * error

            # Fixing first set of weights
            for j in range(0, 20):

                if j < 10:

                    self.weights[0][j] += partialError * partialPrediction * inputs[i][j % 10] * .01

                else:

                    self.weights[0][j] += partialError * partialPrediction2 * inputs[i][j % 10] * .01
