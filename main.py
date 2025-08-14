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
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices based on trained XGBoost model.",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders/maps from models folder
MODEL_PATH = './models/'  # Matches your structure
try:
    model = joblib.load(os.path.join(MODEL_PATH, 'accurate_car_price_model.pkl'))
    label_encoders = joblib.load(os.path.join(MODEL_PATH, 'label_encoders.pkl'))
    name_to_code_maps = joblib.load(os.path.join(MODEL_PATH, 'name_to_code_maps.pkl'))
    code_to_name_maps = joblib.load(os.path.join(MODEL_PATH, 'code_to_name_maps.pkl'))
except Exception as e:
    raise RuntimeError(f"Failed to load model files from {MODEL_PATH}: {str(e)}")

# Model config
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

# Known values for validation
KNOWN_FUEL_CODES = [8, 9, 2561]
KNOWN_TRANS_CODES = [780, 781]

# Extract engine_size
def extract_engine_size(version):
    match = re.search(r'(\d+\.\d+)', version)
    return float(match.group(1)) if match else 1.5  # Default median

# Input schema
class PredictionInput(BaseModel):
    brand: str
    model: str
    year: int
    transmission_code: int
    version: str
    mileage: float
    fuel_type_code: int

# Serve frontend
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse('static/index.html')

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=dict)
async def predict(input_data: PredictionInput):
    try:
        input_dict = input_data.dict(exclude_none=True)
        
        # Map names to codes (flexible for model: try with/without brand prefix)
        brand_lower = input_dict['brand'].lower()
        input_dict['brand_code'] = name_to_code_maps['brand_to_code'].get(brand_lower, -1)
        if input_dict['brand_code'] == -1:
            raise HTTPException(status_code=400, detail=f"Invalid brand: {input_dict['brand']}")

        model_lower = input_dict['model'].lower()
        input_dict['model_code'] = name_to_code_maps['model_to_code'].get(model_lower, name_to_code_maps['model_to_code'].get(f"{brand_lower} {model_lower}", -1))
        if input_dict['model_code'] == -1:
            raise HTTPException(status_code=400, detail=f"Invalid model: {input_dict['model']} for brand {input_dict['brand']}")

        year_str = str(input_dict['year'])
        input_dict['year_code'] = name_to_code_maps['year_to_code'].get(year_str, -1)
        if input_dict['year_code'] == -1:
            raise HTTPException(status_code=400, detail=f"Invalid year: {input_dict['year']}")

        # Validation (only for fuel and transmission)
        if input_dict['fuel_type_code'] not in KNOWN_FUEL_CODES:
            raise HTTPException(status_code=400, detail=f"Invalid fuel_type_code. Known: {KNOWN_FUEL_CODES}")
        if input_dict['transmission_code'] not in KNOWN_TRANS_CODES:
            raise HTTPException(status_code=400, detail=f"Invalid transmission_code. Known: {KNOWN_TRANS_CODES}")

        # Derived features
        input_dict['age'] = CURRENT_YEAR - input_dict['year']
        input_dict['mileage_per_year'] = input_dict['mileage'] / max(input_dict['age'], 1)
        input_dict['age_squared'] = input_dict['age'] ** 2
        input_dict['engine_size'] = extract_engine_size(input_dict['version'])

        # Prepare input DataFrame
        model_input = {feature: input_dict.get(feature, 0) for feature in model_features}
        input_df = pd.DataFrame([model_input])
        logger.info(f"Input DF: {input_df}")

        # Encode (use -1 for unseen values without raising error)
        for col in cat_cols:
            value = str(input_df[col].iloc[0])
            if value in label_encoders[col].classes_:
                input_df[col] = label_encoders[col].transform([value])[0]
            else:
                logger.warning(f"Unseen {col} value: {value}. Using default -1.")
                input_df[col] = -1

        # Log transforms
        input_df['mileage'] = np.log1p(input_df['mileage'])
        input_df['mileage_per_year'] = np.log1p(input_df['mileage_per_year'])
        input_df['engine_size'] = np.log1p(input_df['engine_size'])

        # Predict
        pred_log = model.predict(input_df)[0]
        pred_price = float(np.expm1(pred_log))

        return {"predicted_price": pred_price}
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
