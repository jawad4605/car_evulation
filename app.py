from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib
import pandas as pd
from typing import Optional

app = FastAPI(title="Car Price Prediction API", description="API for predicting car prices based on trained XGBoost model.")

# Add CORS middleware to fix 405 on OPTIONS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for testing; restrict in production (e.g., ["http://localhost"])
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)

# Load the saved model and encoders/maps using joblib for .pkl (ensure files are in the same directory)
model = joblib.load('car_price_xgboost_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
name_to_code_maps = joblib.load('name_to_code_maps.pkl')

# Define model features and cat_cols at top
model_features = ['brand_code', 'model_code', 'year_code', 'transmission_code', 'fuel_type_code', 'version', 'mileage', 'age', 'mileage_per_year', 'age_squared']
cat_cols = ['brand_code', 'model_code', 'year_code', 'transmission_code', 'fuel_type_code', 'version']

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    brand_code: Optional[int] = None
    brand: Optional[str] = None
    model_code: Optional[int] = None
    model: Optional[str] = None
    year_code: Optional[int] = None
    year: Optional[int] = None
    transmission_code: Optional[int] = None
    version: Optional[str] = None
    mileage: float
    fuel_type_code: Optional[int] = None

# Prediction endpoint
@app.post("/predict", response_model=dict)
def predict(input_data: PredictionInput):
    input_dict = input_data.dict(exclude_none=True)
    
    # Map names to codes if codes are missing
    if 'brand_code' not in input_dict and 'brand' in input_dict:
        brand_lower = input_dict['brand'].lower()
        input_dict['brand_code'] = name_to_code_maps['brand_to_code'].get(brand_lower, -1)
    if 'model_code' not in input_dict and 'model' in input_dict:
        model_lower = input_dict['model'].lower()
        input_dict['model_code'] = name_to_code_maps['model_to_code'].get(model_lower, -1)
    if 'year_code' not in input_dict and 'year' in input_dict:
        year_str = str(input_dict['year'])
        input_dict['year_code'] = name_to_code_maps['year_to_code'].get(year_str, -1)
    
    # Compute 'age' if 'year' is provided
    CURRENT_YEAR = 2025
    if 'year' in input_dict:
        input_dict['age'] = CURRENT_YEAR - input_dict['year']
    else:
        raise HTTPException(status_code=400, detail="Missing 'year' for calculating 'age'.")
    
    # Compute 'mileage_per_year' if mileage and age are available
    if 'mileage' in input_dict:
        input_dict['mileage_per_year'] = input_dict['mileage'] / max(input_dict['age'], 1)
    
    # Compute 'age_squared'
    input_dict['age_squared'] = input_dict['age'] ** 2
    
    # Prepare input DataFrame with defaults for missing optional fields
    model_input = {feature: input_dict.get(feature, 0) for feature in model_features}
    input_df = pd.DataFrame([model_input])
    
    # Encode categoricals
    for col in cat_cols:
        value = input_df[col].iloc[0]
        if value in label_encoders[col].classes_:
            input_df[col] = label_encoders[col].transform([value])[0]
        else:
            input_df[col] = -1  # Unknown category handling
    
    # Log transform continuous features
    input_df['mileage'] = np.log1p(input_df['mileage'])
    input_df['mileage_per_year'] = np.log1p(input_df['mileage_per_year'])
    
    # Predict using the loaded XGBRegressor model (no DMatrix needed for predict)
    pred_log = model.predict(input_df)[0]
    pred_price = float(np.expm1(pred_log))  # Convert to float to fix serialization

    return {"predicted_price": pred_price}

# Run the API (use: uvicorn filename:app --reload for local dev)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)