import os
import requests
import json
import numpy as np
from dotenv import load_dotenv

from NeuralNetwork import NeuralNetwork

def grab_data():
    URL = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=" + os.environ["API_KEY"]

    res = requests.get(URL)

    f = open("data/tmp.dat", "w")
    f.write(json.dumps(res.json()))
    f.close()

def load_data():
    pass    

def main():
    load_dotenv()
    
    #grab_data()

    #bot = NeuralNetwork(3, 2, 1, 0.01)

    #bot.train([[1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 1]], [[1], [1], [0], [0], [1]], 2000, 5)

    #print("")

    #bot.forward([1, 0, 1])
    #print("Guess %s" % bot.outputs)

main()
