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
    f = open("data/tmp.dat", "r")
    data = f.read()
    f.close()

    data = json.loads(data)

    return data["Time Series (5min)"]
    

def main():
    load_dotenv()
    
    #grab_data()
    #data = load_data()

    bot = NeuralNetwork(3, 3, 2, 0.01)

    for i in range(0, 100):
        bot.forward(np.array([1, 0, 1]))
        bot.backprop(np.array([1, 1]))

main()
