import os
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

    for i in range(0,1000):
        bot.backpropagate(correct)
        bot.forward(inputs)

    print("")
    print("Correct %s" % correct)
    print("Guess %s" % bot.outputs)

main()
