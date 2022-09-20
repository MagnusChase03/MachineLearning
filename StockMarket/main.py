from StockMarketNerualNetwork import StockMarketNerualNetwork
import json

# Grab all days worth of data and normalize it
def readData(data, date):

    inputs = []

    keys = ['1. open', '2. high', '3. low', '4. close', '5. volume']
    for key in keys:

        # Normalize by / 10
        inputs.append(float(data['Time Series (Daily)'][date][key]) / 100)

    return inputs

def main():

    # Read data
    IBMfile = open('./data/2022-09-17-IBM.json', 'r')
    IBMdata = IBMfile.read()
    data = json.loads(IBMdata)

    # Get inputs for nerual network test
    inputs = readData(data, '2022-09-15')
    dayTwo = readData(data, '2022-09-14')
    
    for element in dayTwo:
        inputs.append(element)

    nerualNetwork = StockMarketNerualNetwork()
    prediction = nerualNetwork.predict(inputs)

    print("Predicted %f" % prediction)
    for i in range(0, 1000):
        nerualNetwork.train([inputs], [12.75300])

    prediction = nerualNetwork.predict(inputs)

    print("Predicted %f" % prediction)
    
main()