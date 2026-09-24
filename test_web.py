import requests

url = "http://127.0.0.1:5000/predict"

tests = [
    {
        "state": "Assam",
        "crop": "Arecanut",
        "area": 50,
        "fertilizer": 100,
        "pesticide": 10
    },
    {
        "state": "Karnataka",
        "crop": "Arhar/Tur",
        "area": 100,
        "fertilizer": 200,
        "pesticide": 20
    },
    {
        "state": "Kerala",
        "crop": "Tapioca",
        "area": 20,
        "fertilizer": 50,
        "pesticide": 5
    }
]

for i, data in enumerate(tests):
    print(f"\n--- WEB TEST {i+1} ---")
    response = requests.post(url, json=data)
    result = response.json()
    if 'error' in result:
        print("Error:", result['error'])
    else:
        print(f"Recommended: {result['recommended_season']}")
        print(f"Yield: {result['yield']} t/ha")
        print(f"Production: {result['production']} t")
        print(f"Comparisons: {result['season_comparison']}")
