from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import joblib
import uvicorn
from ml.data import process_data
import subprocess
import os

# Check if API_LIVE environment variable is set
if os.getenv("API_LIVE"):
    # Run dvc pull to fetch the latest data and model artifacts
    try:
        subprocess.run(["dvc", "pull"], check=True)
        print("dvc pull executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during dvc pull: {e}")
        raise RuntimeError(
            "Failed to execute dvc pull. "
            "Please check your DVC setup "
            )


# Load the model and preprocessing artifacts
model = joblib.load("starter/model/model.joblib")
encoder = joblib.load("starter/model/encoder.joblib")
lb = joblib.load("starter/model/lb.joblib")

# Define the FastAPI app
app = FastAPI()

# Pydantic model for POST request body
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

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
                "native-country": "United-States",
            }
        }

# Root GET endpoint
@app.get("/")
async def welcome():
    return {"message": "Welcome to the Census Income Prediction API!"}

# POST endpoint for model inference
@app.post("/predict")
async def predict(data: CensusData):
    try:
        # Convert the input data into a DataFrame
        input_data = data.dict(by_alias=True)
        df = pd.DataFrame([input_data])

        # Preprocess the data
        categorical_features = [
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
            df,
            categorical_features=categorical_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Make a prediction
        prediction = model.predict(X)
        print(prediction)
        predicted_label = lb.inverse_transform(prediction)[0]

        return {"prediction": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("starter.main:app", host="0.0.0.0", port=8000, reload=True)