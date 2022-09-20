import requests
import json
import os
from dotenv import load_dotenv
from datetime import date

# Load API Key
load_dotenv()

# Grab stock data
company = 'IBM'

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + company + '&apikey=' + os.getenv('API_KEY')
response = requests.get(url)
data = response.json()

# Write to file
f = open('./data/' + str(date.today()) + '-' + company + '.json', 'w')
f.write(json.dumps(data, indent=4))