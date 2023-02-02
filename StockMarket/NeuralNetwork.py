import numpy as np
import random

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
        
