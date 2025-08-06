import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib  # For saving encoders/maps
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm  # For lowess smoothing

# Path to your CSV
CSV_PATH = 'Data_set/abc_car_valuation_data.csv'  # Update if needed

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}. Please update the path.")
    exit(1)

# Load and clean data
df = pd.read_csv(CSV_PATH, quotechar='"')
df.columns = df.columns.str.strip().str.replace('"', '')

print("Loaded columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head(5))
print("\nData types:")
print(df.dtypes)
print("\nShape:", df.shape)  # Should show ~178k rows

# Handle target
if 'actual_price' not in df.columns:
    raise KeyError("Column 'actual_price' not found.")
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df = df.dropna(subset=['actual_price'])

# Required features for model (codes, version, mileage, age, mileage_per_year, age_squared for non-linear depreciation)
model_features = ['brand_code', 'model_code', 'year_code', 'transmission_code', 'fuel_type_code', 'version', 'mileage', 'age', 'mileage_per_year', 'age_squared']

# All columns for display/understanding (including names)
all_columns = ['brand_code', 'brand', 'model_code', 'model', 'year_code', 'year', 'transmission_code', 'version', 'mileage', 'fuel_type_code']

# Add 'age' feature: current year (2025) - vehicle year
CURRENT_YEAR = 2025
if 'year' not in df.columns:
    raise KeyError("Column 'year' not found for calculating age.")
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['age'] = CURRENT_YEAR - df['year']

# Add new interaction features: mileage_per_year and age_squared (for non-linear age effect)
df['mileage_per_year'] = df['mileage'] / df['age'].replace(0, 1)  # Avoid division by zero for new cars
df['age_squared'] = df['age'] ** 2  # Quadratic term for better depreciation modeling

for col in model_features[:-3]:  # Exclude new features as they're derived
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    if col != 'version':  # Convert codes to int, handle NaNs
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
df = df.dropna(subset=model_features + ['actual_price'])

# Define luxury brands for higher weighting (based on common premium brands; adjust as needed)
luxury_brands = ['Volvo', 'Audi', 'Bmw', 'Mercedes', 'Porsche', 'Jaguar', 'Land rover', 'Lexus']  # Example list
df['weight'] = np.where(df['brand'].isin(luxury_brands), 2.0, 1.0)  # Double weight for luxury to reduce underprediction

# Create mappings from names to codes for prediction (if user inputs names)
name_to_code_maps = {
    'brand_to_code': dict(zip(df['brand'].str.lower(), df['brand_code'])),
    'model_to_code': dict(zip(df['model'].str.lower(), df['model_code'])),
    'year_to_code': dict(zip(df['year'].astype(str), df['year_code'])),
    # Assuming transmission and fuel have no name equivalents; if they do, add here
}

# Encode categoricals for model (XGBoost handles ints/strings, but LabelEncoder for safety)
label_encoders = {}
cat_cols = ['brand_code', 'model_code', 'year_code', 'transmission_code', 'fuel_type_code', 'version']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Handle strings like version
    label_encoders[col] = le

# Optional: Log transform target/mileage if skewed (improves accuracy)
df['actual_price'] = np.log1p(df['actual_price'])  # Log for positive skew
df['mileage'] = np.log1p(df['mileage'])
df['mileage_per_year'] = np.log1p(df['mileage_per_year'])  # Also log the new feature

# Split data, including weights
X = df[model_features]
y = df['actual_price']
weights = df['weight']
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Optimize with expanded GridSearchCV for better hyperparameters
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', subsample=0.8, colsample_bytree=0.8, seed=42)
param_grid = {
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.005, 0.01, 0.02],
    'n_estimators': [800, 1000, 1200, 1400],
    'reg_lambda': [0.5, 1, 1.5]  # Added L2 regularization for better generalization
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train, sample_weight=w_train)

# Best model from grid search
model = grid_search.best_estimator_
print(f"Best parameters from GridSearch: {grid_search.best_params_}")

# Evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nTest MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")  # Aim >0.8 for good fit

# Predict on full dataset to compare actual vs predicted
df['predicted_log'] = model.predict(df[model_features])
df['predicted'] = np.expm1(df['predicted_log'])
df['actual_original'] = np.expm1(df['actual_price'])  # Add original actual for comparison

# Plot actual vs predicted (scatter plot on original scale)
plt.figure(figsize=(10, 6))
plt.scatter(df['actual_original'], df['predicted'], alpha=0.3, label='Predictions')
plt.plot([df['actual_original'].min(), df['actual_original'].max()], [df['actual_original'].min(), df['actual_original'].max()], color='red', linestyle='--', label='Perfect Fit')
# Add smoothed trend line using lowess
lowess = sm.nonparametric.lowess(df['predicted'], df['actual_original'], frac=0.1)
plt.plot(lowess[:, 0], lowess[:, 1], color='blue', label='Trend Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices on Full Dataset (Scatter with Trend Line)')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted_scatter_trend.png')  # Save for viewing
plt.show()
print("Scatter plot with trend line saved as 'actual_vs_predicted_scatter_trend.png'.")

# Additional line graph: Sort by actual price and plot lines for actual and predicted
sorted_df = df.sort_values('actual_original').reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.plot(sorted_df.index, sorted_df['actual_original'], label='Actual Price (Sorted)', color='green')
plt.plot(sorted_df.index, sorted_df['predicted'], label='Predicted Price', color='orange', linestyle='--')
plt.xlabel('Sorted Index (by Actual Price)')
plt.ylabel('Price')
plt.title('Line Graph: Sorted Actual vs Predicted Prices')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted_line_graph.png')  # Save for viewing
plt.show()
print("Line graph saved as 'actual_vs_predicted_line_graph.png'. Open it to view the comparison.")

# Metrics on full data
mse_full = mean_squared_error(df['actual_original'], df['predicted'])
r2_full = r2_score(df['actual_original'], df['predicted'])
print(f"\nFull Dataset MSE (original scale): {mse_full:.4f}")
print(f"Full Dataset R² (original scale): {r2_full:.4f}")

# Save model and encoders/maps
joblib.dump(model, 'car_price_xgboost_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(name_to_code_maps, 'name_to_code_maps.pkl')
print("Model and mappings saved.")

# Prediction function (accepts all values, uses only model_features, displays all for understanding)
def predict_price(input_dict):
    # input_dict: dict with keys like 'brand_code', 'brand', 'model_code', etc.
    
    # Map names to codes if codes are missing
    if 'brand_code' not in input_dict and 'brand' in input_dict:
        brand_lower = input_dict['brand'].lower()
        input_dict['brand_code'] = name_to_code_maps['brand_to_code'].get(brand_lower, -1)  # -1 for unknown
    if 'model_code' not in input_dict and 'model' in input_dict:
        model_lower = input_dict['model'].lower()
        input_dict['model_code'] = name_to_code_maps['model_to_code'].get(model_lower, -1)
    if 'year_code' not in input_dict and 'year' in input_dict:
        year_str = str(input_dict['year'])
        input_dict['year_code'] = name_to_code_maps['year_to_code'].get(year_str, -1)
    
    # Compute 'age' if 'year' is provided
    if 'year' in input_dict:
        input_dict['age'] = CURRENT_YEAR - input_dict['year']
    else:
        raise ValueError("Missing 'year' for calculating 'age'.")
    
    # Compute 'mileage_per_year' if mileage and age are available
    if 'mileage' in input_dict:
        input_dict['mileage_per_year'] = input_dict['mileage'] / max(input_dict['age'], 1)
    
    # Compute 'age_squared'
    input_dict['age_squared'] = input_dict['age'] ** 2
    
    # Prepare model input DataFrame (only model_features)
    model_input = {}
    for feature in model_features:
        if feature in input_dict:
            model_input[feature] = [input_dict[feature]]
        else:
            raise ValueError(f"Missing required model feature: {feature}")
    
    input_df = pd.DataFrame(model_input)
    
    # Encode categoricals using saved encoders
    for col in cat_cols:
        if input_df[col].iloc[0] in label_encoders[col].classes_:
            input_df[col] = label_encoders[col].transform(input_df[col])
        else:
            input_df[col] = -1  # Unknown
    
    # Log mileage and mileage_per_year
    input_df['mileage'] = np.log1p(input_df['mileage'])
    input_df['mileage_per_year'] = np.log1p(input_df['mileage_per_year'])
    
    # Predict
    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)  # Inverse log
    
    # Display all input values for understanding
    print("\nInput Details:")
    for key, value in input_dict.items():
        print(f"{key}: {value}")
    return pred_price

# Interactive prediction (now accepts all values)
print("\nEnter values for prediction (or 'exit' to quit). You can provide codes or names; missing codes will be mapped from names if possible.")
while True:
    try:
        input_dict = {}
        for col in all_columns:
            value = input(f"{col} (or skip with enter): ")
            if value.lower() == 'exit':
                raise KeyboardInterrupt  # To break loop
            if value:  # Only add if provided
                try:
                    input_dict[col] = float(value) if '.' in value else int(value)  # Handle numeric
                except ValueError:
                    input_dict[col] = value  # String
        
        if not input_dict:
            continue
        
        price = predict_price(input_dict)
        print(f"Predicted price: ${price:.2f}\n")
    except KeyboardInterrupt:
        break
    except ValueError as e:
        print(f"Invalid input: {e}")