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

def entry_to_array(data):

    array = np.zeros(5)
    index = 0
    for key in data.keys():
        array[index] = float(data[key])
        index += 1

    return array

def load_dataset(data):
    
    dataset = []
    for key in sorted(data.keys()):
        dataset.append(entry_to_array(data[key]))

    return np.array(dataset)

def main():
    load_dotenv()
    
    #grab_data()
    data = load_data()
    dataset = load_dataset(data)

    bot = NeuralNetwork(5, 6, 5, 0.01)
    bot.train(dataset[:-1], dataset[1:], 100)

    bot.forward(dataset[0])
    print(bot.outputs)
    print(dataset[1])

main()
