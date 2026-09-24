import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "state": "Assam",
    "crop": "Arecanut",
    "season": "Whole Year",
    "year": 2020,
    "area": 50,
    "fertilizer": 100,
    "pesticide": 10
}

response = requests.post(url, json=data)
print(response.json())
