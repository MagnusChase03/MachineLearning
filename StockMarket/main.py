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
    data = load_data()

    bot = NeuralNetwork(5, 6, 5, 0.00001)

    dataSets = []
    for key in data.keys():
        index = 0
        dataSet = np.zeros(5)
        for label in data[key].keys():
            dataSet[index] = data[key][label]
        
        dataSets.append(dataSet)

    bot.train(dataSets[0:2], dataSets[1:3], 3000, 2)

main()
