import os
from dotenv import load_dotenv

from NeuralNetwork import NeuralNetwork

def main():
    load_dotenv()

    bot = NeuralNetwork(5, 6, 5)
    print(bot.weights)

main()
