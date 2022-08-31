import requests

news = {'title': 'U.S. Senate opposition to Obamacare repeal bill grows'}

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=news)

print(response.json())
