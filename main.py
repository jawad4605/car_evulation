# main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import re
import datetime
from typing import Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices based on trained XGBoost model.",
    version="1.0.0"
)

# Serve static files (including index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders/maps
try:
    model = joblib.load('accurate_car_price_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    name_to_code_maps = joblib.load('name_to_code_maps.pkl')
    code_to_name_maps = joblib.load('code_to_name_maps.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

# Known values from training for validation
unique_fuel = [8, 9, 2561]  # Adjust based on your actual unique values
unique_trans = [780, 781]  # Adjust based on your actual unique values
unique_version_patterns = r'\d|\btdi\b|\btfsi\b|\bs line\b|\bquattro\b|\bsport\b|\bdesign\b|\be-tron\b'

# Model configuration
model_features = [
    'brand_code', 'model_code', 'year_code', 'transmission_code',
    'fuel_type_code', 'version', 'mileage', 'age',
    'mileage_per_year', 'age_squared', 'engine_size'
]
cat_cols = [
    'brand_code', 'model_code', 'year_code',
    'transmission_code', 'fuel_type_code', 'version'
]
CURRENT_YEAR = datetime.datetime.now().year

# Function to extract engine_size
def extract_engine_size(version):
    match = re.search(r'(\d+\.\d+)', version)
    return float(match.group(1)) if match else np.nan

# Input schema
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

# Serve frontend at root
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse('static/index.html')

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=dict)
async def predict(input_data: PredictionInput):
    try:
        input_dict = input_data.dict(exclude_none=True)
       
        # Map names to codes
        if 'brand_code' not in input_dict and 'brand' in input_dict:
            brand_lower = input_dict['brand'].lower()
            input_dict['brand_code'] = name_to_code_maps['brand_to_code'].get(brand_lower, -1)
        elif 'brand' not in input_dict and 'brand_code' in input_dict:
            input_dict['brand'] = code_to_name_maps['code_to_brand'].get(input_dict['brand_code'], 'Unknown')

        if 'model_code' not in input_dict and 'model' in input_dict:
            model_lower = input_dict['model'].lower()
            input_dict['model_code'] = name_to_code_maps['model_to_code'].get(model_lower, -1)
        elif 'model' not in input_dict and 'model_code' in input_dict:
            input_dict['model'] = code_to_name_maps['code_to_model'].get(input_dict['model_code'], 'Unknown')

        if 'year_code' not in input_dict and 'year' in input_dict:
            year_str = str(input_dict['year'])
            input_dict['year_code'] = name_to_code_maps['year_to_code'].get(year_str, -1)
        elif 'year' not in input_dict and 'year_code' in input_dict:
            input_dict['year'] = code_to_name_maps['code_to_year'].get(input_dict['year_code'], -1)
            if input_dict['year'] == -1:
                raise HTTPException(status_code=400, detail="Invalid year_code provided")

        if 'year' not in input_dict:
            raise HTTPException(status_code=400, detail="Either year or year_code is required")

        # Validation
        if 'fuel_type_code' in input_dict and input_dict['fuel_type_code'] not in unique_fuel:
            raise HTTPException(status_code=400, detail=f"Invalid fuel_type_code: {input_dict['fuel_type_code']}. Known: {unique_fuel}")
        if 'transmission_code' in input_dict and input_dict['transmission_code'] not in unique_trans:
            raise HTTPException(status_code=400, detail=f"Invalid transmission_code: {input_dict['transmission_code']}. Known: {unique_trans}")
        if 'mileage' in input_dict and not (0 <= input_dict['mileage'] <= 500000):
            raise HTTPException(status_code=400, detail=f"Invalid mileage: {input_dict['mileage']}. Must be 0-500,000 km.")
        if 'version' in input_dict and not re.search(unique_version_patterns, input_dict['version'], re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Invalid version: Lacks typical engine/trim patterns.")

        # Calculate derived features
        input_dict['age'] = CURRENT_YEAR - input_dict['year']
        input_dict['mileage_per_year'] = input_dict['mileage'] / max(input_dict['age'], 1)
        input_dict['age_squared'] = input_dict['age'] ** 2
        input_dict['engine_size'] = extract_engine_size(input_dict.get('version', '')) or np.nanmedian([1.0])  # Use a default median if NaN

        # Prepare input DataFrame
        model_input = {feature: input_dict.get(feature, -1) for feature in model_features}
        input_df = pd.DataFrame([model_input])

        # Encode categoricals
        for col in cat_cols:
            value = input_df[col].iloc[0]
            if value in label_encoders[col].classes_:
                input_df[col] = label_encoders[col].transform([value])[0]
            else:
                input_df[col] = -1

        # Transform features
        input_df['mileage'] = np.log1p(input_df['mileage'])
        input_df['mileage_per_year'] = np.log1p(input_df['mileage_per_year'])
        input_df['engine_size'] = np.log1p(input_df['engine_size'])

        # Make prediction
        pred_log = model.predict(input_df)[0]
        pred_price = float(np.expm1(pred_log))

        return {"predicted_price": pred_price}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
