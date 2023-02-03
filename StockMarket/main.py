import os
import numpy as np
from dotenv import load_dotenv

from NeuralNetwork import NeuralNetwork

def main():
    load_dotenv()

    bot = NeuralNetwork(5, 6, 5)
    bot.forward(np.random.rand(5))
    print(bot.hiddenLayer[0])
    print(bot.hiddenLayer[1])
    print(bot.outputs)

main()
