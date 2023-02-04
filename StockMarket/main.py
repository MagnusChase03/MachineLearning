import os
import numpy as np
from dotenv import load_dotenv

from NeuralNetwork import NeuralNetwork

def main():
    load_dotenv()

    bot = NeuralNetwork(3, 2, 1)

    bot.train([[1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1]], [[1], [1], [0], [0], [1]], 2000)

    print("")

    bot.forward([1, 0, 1])
    print("Guess %s" % bot.outputs)

main()
