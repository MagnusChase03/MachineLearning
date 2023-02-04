import os
import time
import numpy as np
from dotenv import load_dotenv

from NeuralNetwork import NeuralNetwork

def main():
    load_dotenv()

    bot = NeuralNetwork(5, 6, 5)

    inputs = np.random.rand(5)
    correct = np.random.rand(5)
    bot.forward(inputs)

    print("Correct %s" % correct)
    print("Guess %s" % bot.outputs)

    bot.train([inputs], [correct], 200)

    print("")
    print("Correct %s" % correct)
    print("Guess %s" % bot.outputs)

main()
