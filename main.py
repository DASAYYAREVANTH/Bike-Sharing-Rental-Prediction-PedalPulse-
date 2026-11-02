
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = Path("bike_rental_model.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "bike_rental_model.pkl not found!"
    )

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
model_name = artifact["model_name"]
feature_columns = artifact["feature_columns"]

date_min = pd.Timestamp(artifact["date_min"]).date()
date_max = pd.Timestamp(artifact["date_max"]).date()

print("Model loaded:", model_name)

# ==========================================
# 2. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Bike Rental Demand Prediction API",
    description="Predict bike rental demand using Machine Learning",
    version="1.0"
)

# ==========================================
# 3. DEFINE INPUT DATA
# ==========================================

class RentalInput(BaseModel):

    rental_date: date

    hour: int = Field(ge=0, le=23)

    season: Literal[
        "springer", "summer", "fall", "winter"
    ]

    holiday: Literal["No", "Yes"]

    workingday: Literal[
        "No work", "Working Day"
    ]

    weathersit: Literal[
        "Clear", "Mist", "Light Snow", "Heavy Rain"
    ]

    temp: float = Field(ge=0, le=1)
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)


# ==========================================
# 4. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Bike Rental Prediction API is running",
        "model": model_name
    }


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: RentalInput):

    # Validate supported historical dates
    if not date_min <= data.rental_date <= date_max:
        raise HTTPException(
            status_code=422,
            detail=f"Select a date between {date_min} and {date_max}"
        )

    rental_date = data.rental_date

    # Dataset weekday encoding: Sunday = 0
    weekday = (rental_date.weekday() + 1) % 7

    month = rental_date.month

    # ======================================
    # FEATURE ENGINEERING
    # ======================================

    input_data = {

        "season": data.season,

        "yr": rental_date.year,

        "holiday": data.holiday,

        "workingday": data.workingday,

        "weathersit": data.weathersit,

        "temp": data.temp,

        "atemp": data.atemp,

        "hum": data.hum,

        "windspeed": data.windspeed,

        "hour_sin": np.sin(
            2 * np.pi * data.hour / 24
        ),

        "hour_cos": np.cos(
            2 * np.pi * data.hour / 24
        ),

        "weekday_sin": np.sin(
            2 * np.pi * weekday / 7
        ),

        "weekday_cos": np.cos(
            2 * np.pi * weekday / 7
        ),

        "month_sin": np.sin(
            2 * np.pi * month / 12
        ),

        "month_cos": np.cos(
            2 * np.pi * month / 12
        ),

        "day_of_month": rental_date.day,

        "day_of_year": rental_date.timetuple().tm_yday
    }

    # Create DataFrame in exact training feature order
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    # ======================================
    # GENERATE PREDICTION
    # ======================================

    try:

        prediction = model.predict(input_df)[0]

        prediction = max(0, round(float(prediction)))

        return {
            "model": model_name,
            "predicted_rentals": prediction,
            "date": str(rental_date),
            "hour": data.hour,
            "message": "Prediction generated successfully"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
