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

def dataToArray(data):

    array = np.zeros(5)
    index = 0
    for key in data.keys():
        array[index] = float(data[key])
        index += 1

    return array

def main():
    load_dotenv()
    
    #grab_data()
    data = load_data()

    bot = NeuralNetwork(5, 6, 5, 0.01)

    inputs = dataToArray(data["2023-02-06 07:15:00"])
    outputs = dataToArray(data["2023-02-06 07:45:00"])
    inputs2 = dataToArray(data["2023-02-06 07:45:00"])
    outputs2 = dataToArray(data["2023-02-06 08:05:00"])
    bot.train([inputs, inputs2], [outputs, outputs2], 1000)

main()
