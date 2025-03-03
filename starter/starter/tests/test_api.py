import pytest
from fastapi.testclient import TestClient
from starter.main import app  

# Create a test client
client = TestClient(app)

# Test GET request to the root endpoint
def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API!"}

# Test POST request with valid input data
def test_predict_high_income():
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
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert ">50K" in response.json()["prediction"]

# Test POST request with another valid input data
def test_predict_low_income():
    test_data = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 83311,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 13,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=test_data)
    print(response.json())
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "<=50K" in response.json()["prediction"]