# Unit test cases
# Run tests with: `python -m pytest test_main.py -v`

from fastapi.testclient import TestClient
import pytest
from main import app

client = TestClient(app)

@pytest.fixture
def data_more_than_50k():
    """Fixture for data that should predict income >50K"""
    return {
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
        "hours-per-week": 50,
        "native-country": "United-States"
    }

@pytest.fixture
def data_less_than_50k():
    """Fixture for data that should predict income <=50K"""
    return {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

@pytest.fixture
def data_missing_field():
    """Fixture for data with missing required field"""
    return {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
# missing fnlgt

@pytest.fixture
def data_invalid_type():
    """Fixture for data with invalid type"""
    return {
        "age": "not-a-number",  
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
# invalid type for age


def test_get_root():
    """Test the root endpoint GET request"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}

def test_post_predict_less_than_50k(data_less_than_50k):
    """Test prediction endpoint with data that should predict income <= 50K"""
    response = client.post("/predict", json=data_less_than_50k)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert response.json()["prediction"] in ["<=50K", ">50K"]

def test_post_predict_more_than_50k(data_more_than_50k):
    """Test prediction endpoint with data that should predict income >50K"""
    response = client.post("/predict", json=data_more_than_50k)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert response.json()["prediction"] in ["<=50K", ">50K"]

def test_post_predict_missing_field(data_missing_field):
    """Test prediction endpoint with missing required field"""
    response = client.post("/predict", json=data_missing_field)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()
    assert "fnlgt" in str(response.json()["detail"])

def test_post_predict_invalid_type(data_invalid_type):
    """Test prediction endpoint with invalid data type"""
    response = client.post("/predict", json=data_invalid_type)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()
    assert "age" in str(response.json()["detail"])

def test_post_predict_empty_payload():
    """Test prediction endpoint with empty payload"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()