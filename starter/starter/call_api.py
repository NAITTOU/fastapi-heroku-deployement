import requests

# Define the API URL
API_URL = "https://fastapi-heroku-deployement.onrender.com/predict"

# Define the input data for the POST request
test_data = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5178,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send the POST request
try:
    response = requests.post(API_URL, json=test_data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    print("Response from API:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error making POST request: {e}")