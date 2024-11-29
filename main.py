from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Cargar el modelo
model_pipeline = joblib.load('pipeline.joblib')

class SalaryPredictionInput(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    employee_residence: str
    company_location: str
    company_size: str

@app.post("/predict")
def predict_salary(input_data: SalaryPredictionInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    prediction = model_pipeline.predict(input_df)[0]
    return {"predicted_salary": prediction}
