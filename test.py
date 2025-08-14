import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json
import re
import seaborn as sns
from scipy import stats

# Paths to JSON files (update as needed)
BRANDS_JSON = 'json_files/unique_brands.json'
MODELS_JSON = 'json_files/unique_models.json'
YEARS_JSON = 'json_files/unique_years.json'
MAIN_DATA_JSON = 'json_files/updated_car_valuation_data.json'

# Load unique mappings from JSON
def load_unique_data(json_path, code_key, name_key):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if 'data' in item:
            records = item['data']
            break
    else:
        raise ValueError(f"No 'data' found in {json_path}")
    name_to_code = {record[name_key].lower(): record[code_key] for record in records}
    code_to_name = {record[code_key]: record[name_key] for record in records}
    return name_to_code, code_to_name

brand_to_code, code_to_brand = load_unique_data(BRANDS_JSON, 'brand_code', 'brand')
model_to_code, code_to_model = load_unique_data(MODELS_JSON, 'model_code', 'model')
year_to_code_raw, code_to_year_raw = load_unique_data(YEARS_JSON, 'year_code', 'year')
year_to_code = {k: v for k, v in year_to_code_raw.items() if k.isdigit()}
code_to_year = {k: int(v) for k, v in code_to_year_raw.items() if v.isdigit()}

name_to_code_maps = {'brand_to_code': brand_to_code, 'model_to_code': model_to_code, 'year_to_code': year_to_code}
code_to_name_maps = {'code_to_brand': code_to_brand, 'code_to_model': code_to_model, 'code_to_year': code_to_year}

# Load main data
with open(MAIN_DATA_JSON, 'r') as f:
    data = json.load(f)
for item in data:
    if 'data' in item:
        records = item['data']
        break
else:
    raise ValueError(f"No 'data' found in {MAIN_DATA_JSON}")
df = pd.DataFrame(records)
df.columns = df.columns.str.strip().str.replace('"', '')

# Preprocessing and Validation
required_cols = ['brand_code', 'model_code', 'year_code', 'transmission_code', 'fuel_type_code', 'version', 'mileage', 'actual_price', 'year']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise KeyError(f"Missing columns: {missing_cols}")

# Convert numerics, drop invalid
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=required_cols)

# Feature Validation (extract uniques for checks)
unique_fuel = df['fuel_type_code'].unique().tolist()  # e.g., [8, 9, 2561]
unique_trans = df['transmission_code'].unique().tolist()  # e.g., [780, 781]
unique_version_patterns = r'\d|\btdi\b|\btfsi\b|\bs line\b|\bquattro\b|\bsport\b|\bdesign\b|\be-tron\b'  # Simplified regex

# Validate data rows
df = df[df['fuel_type_code'].isin(unique_fuel)]
df = df[df['transmission_code'].isin(unique_trans)]
df = df[df['mileage'].between(0, 500000)]
df = df[df['version'].str.contains(unique_version_patterns, regex=True, case=False, na=False)]

# Feature Engineering
CURRENT_YEAR = 2025
df['age'] = CURRENT_YEAR - df['year']
df['mileage_per_year'] = df['mileage'] / df['age'].clip(lower=1)
df['age_squared'] = df['age'] ** 2

# Extract engine_size from version
def extract_engine_size(version):
    match = re.search(r'(\d+\.\d+)', version)
    return float(match.group(1)) if match else np.nan
df['engine_size'] = df['version'].apply(extract_engine_size)
df['engine_size'] = df['engine_size'].fillna(df['engine_size'].median())

model_features = ['brand_code', 'model_code', 'year_code', 'transmission_code', 'fuel_type_code', 'version', 
                  'mileage', 'age', 'mileage_per_year', 'age_squared', 'engine_size']

# Weights for luxury brands
luxury_brands = ['volvo', 'audi', 'bmw', 'mercedes', 'porsche', 'jaguar', 'land rover', 'lexus']
df['brand'] = df.apply(lambda row: code_to_brand.get(row['brand_code'], row.get('brand', 'unknown')), axis=1)
df['weight'] = np.where(df['brand'].str.lower().isin(luxury_brands), 2.0, 1.0)

# Encoding
label_encoders = {}
cat_cols = ['brand_code', 'model_code', 'year_code', 'transmission_code', 'fuel_type_code', 'version']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Log transforms (for skewness)
df['actual_price'] = np.log1p(df['actual_price'])
df['mileage'] = np.log1p(df['mileage'])
df['mileage_per_year'] = np.log1p(df['mileage_per_year'])
df['engine_size'] = np.log1p(df['engine_size'])

# Split
X = df[model_features]
y = df['actual_price']
weights = df['weight']
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Hyperparameter Tuning (reduced grid if large dataset)
param_grid = {
    'max_depth': [3, 5, 7] if len(df) > 100000 else [3, 4, 5, 6, 7, 8],  # Smaller grid for large data
    'learning_rate': [0.01, 0.02, 0.05],
    'n_estimators': [500, 1000, 1500],
    'reg_lambda': [0.5, 1, 2]
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', subsample=0.8, colsample_bytree=0.8, seed=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train, sample_weight=w_train)

# Best model
model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nTest MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Full predictions
df['predicted_log'] = model.predict(df[model_features])
df['predicted'] = np.expm1(df['predicted_log'])
df['actual_original'] = np.expm1(df['actual_price'])

# Graph 1: Scatter Plot with Trend
plt.figure(figsize=(10, 6))
plt.scatter(df['actual_original'], df['predicted'], alpha=0.3, label='Predictions')
plt.plot([df['actual_original'].min(), df['actual_original'].max()], [df['actual_original'].min(), df['actual_original'].max()], color='red', linestyle='--', label='Perfect Fit')
lowess = sm.nonparametric.lowess(df['predicted'], df['actual_original'], frac=0.1)
plt.plot(lowess[:, 0], lowess[:, 1], color='blue', label='Trend Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices (Scatter with Trend)')
plt.legend()
plt.grid(True)
plt.savefig('plot/actual_vs_predicted_scatter.png')
plt.show()

# Graph 2: Residual Plot
residuals = df['actual_original'] - df['predicted']
plt.figure(figsize=(10, 6))
plt.scatter(df['predicted'], residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot (Random around 0 = Good Model)')
plt.grid(True)
plt.savefig('plot/residual_plot.png')
plt.show()

# Graph 3: Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Residuals')
plt.title('Histogram of Residuals (Normal Distribution = Good)')
plt.grid(True)
plt.savefig('plot/residuals_histogram.png')
plt.show()

# Graph 4: QQ Plot for Normality
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals (Straight Line = Normal)')
plt.grid(True)
plt.savefig('plot/qq_plot.png')
plt.show()

# Graph 5: Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=model_features)
plt.title('Feature Importances (Higher = More Impact on Price)')
plt.grid(True)
plt.savefig('plot/feature_importances.png')
plt.show()

# Save model and artifacts
joblib.dump(model, 'model/accurate_car_price_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')
joblib.dump(name_to_code_maps, 'model/name_to_code_maps.pkl')
joblib.dump(code_to_name_maps, 'model/code_to_name_maps.pkl')
print("Model saved.")