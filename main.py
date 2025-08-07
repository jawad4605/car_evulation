from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib
import pandas as pd
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

# Load model and encoders
try:
    model = joblib.load('car_price_xgboost_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    name_to_code_maps = joblib.load('name_to_code_maps.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

# Model configuration
model_features = [
    'brand_code', 'model_code', 'year_code', 'transmission_code',
    'fuel_type_code', 'version', 'mileage', 'age',
    'mileage_per_year', 'age_squared'
]
cat_cols = [
    'brand_code', 'model_code', 'year_code',
    'transmission_code', 'fuel_type_code', 'version'
]
CURRENT_YEAR = 2025

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

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Car Price Prediction API is running"}

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
        if 'model_code' not in input_dict and 'model' in input_dict:
            model_lower = input_dict['model'].lower()
            input_dict['model_code'] = name_to_code_maps['model_to_code'].get(model_lower, -1)
        if 'year_code' not in input_dict and 'year' in input_dict:
            year_str = str(input_dict['year'])
            input_dict['year_code'] = name_to_code_maps['year_to_code'].get(year_str, -1)
        
        # Calculate derived features
        if 'year' in input_dict:
            input_dict['age'] = CURRENT_YEAR - input_dict['year']
        else:
            raise HTTPException(status_code=400, detail="Year is required")
            
        input_dict['mileage_per_year'] = input_dict['mileage'] / max(input_dict['age'], 1)
        input_dict['age_squared'] = input_dict['age'] ** 2
        
        # Prepare input DataFrame
        model_input = {feature: input_dict.get(feature, 0) for feature in model_features}
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
        
        # Make prediction
        pred_log = model.predict(input_df)[0]
        pred_price = float(np.expm1(pred_log))
        
        return {"predicted_price": pred_price}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Redirect root to static index.html
@app.get("/", include_in_schema=False)
async def serve_frontend():
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)