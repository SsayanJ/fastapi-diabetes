"""
Note: you need to run the app from the root folder otherwise the models folder will not be found
- To run the app
$ uvicorn serving.model_as_service.main:app --reload
- To make a prediction from terminal
$ curl -X 'POST' 'http://127.0.0.1:8000/predict_obj' \
  -H 'accept: application/json' -H 'Content-Type: application/json' \
  -d '{ "age": 0, "sex": 0, "bmi": 0, "bp": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0, "s6": 0 }'
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()
model = joblib.load('models/diabetes_model.pkl')
with open("models/model_config.json") as f:
    model_config = json.load(f)

FIELD_LIST = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


class DiabetesInfo(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/")
async def root():
    return {"message": "FastAPI for diabetes evolution prediction"}


@app.get("/model_config")
async def return_model_config():
    return model_config


@app.post('/predict')
async def predict_diabetes_progress(user_info: DiabetesInfo):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level
    user_info = user_info.dict()
    model_input_data = np.array([user_info[field] for field in FIELD_LIST]).reshape(1, -1)
    progression = model.predict(model_input_data)
    print(type(progression))
    return progression[0]


@app.post('/predict_obj')
async def predict_diabetes_progress_batch(diabetes_info: List[DiabetesInfo]):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level
    model_input_data = pd.DataFrame([inf.dict() for inf in diabetes_info])
    progression = model.predict(model_input_data)
    res = {"patient " + str(i): progression[i] for i in range(model_input_data.shape[0])}
    return res
