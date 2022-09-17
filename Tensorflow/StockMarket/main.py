from StockMarketNerualNetwork import StockMarketNerualNetwork
import json

def readData(data, date):

    inputs = []

    keys = ['1. open', '2. high', '3. low', '4. close', '5. volume']
    for key in keys:
        inputs.append(float(data['Time Series (Daily)'][date][key]))

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
    
main()