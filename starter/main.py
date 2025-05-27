# Put the code for your API here. Run the API using: `uvicorn main:app --reload`
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from starter.ml.data import process_data
from starter.ml.model import inference

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and preprocessors
if not os.path.exists("model/model.pkl"):
    raise RuntimeError("Model file not found. Please train the model first.")

model = pickle.load(open("model/model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))
lb = pickle.load(open("model/lb.pkl", "rb"))

# Define the input model with example
class CensusInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
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
        }

@app.get("/")
async def welcome():
    """
    Root endpoint with welcome message
    """
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/predict")
async def predict(data: CensusInput):
    """
    Make predictions with the model
    """
    # Convert input to DataFrame
    input_data = pd.DataFrame(
        {k: [v] for k, v in data.dict(by_alias=True).items()}
    )

    # Process the input data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make prediction
    prediction = inference(model, X)
    prediction_label = lb.inverse_transform(prediction)[0]

    return {
        "prediction": prediction_label,
        "probability": model.predict_proba(X).tolist()[0]
    }