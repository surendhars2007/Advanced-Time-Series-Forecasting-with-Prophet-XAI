# Advanced Time Series Forecasting with Prophet + XAI
# Requirements: prophet, pandas, numpy, matplotlib, shap, scikit-learn, xgboost

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
from xgboost import XGBRegressor

# -----------------------------
# Step 2: Generate Sample Data
# -----------------------------
np.random.seed(42)

# 3 years of daily data
days = pd.date_range(start="2020-01-01", end="2022-12-31")
n = len(days)

# Base trend
trend = np.linspace(10, 50, n)

# Yearly seasonality
yearly = 10 * np.sin(2 * np.pi * np.arange(n)/365)

# Weekly seasonality
weekly = 5 * np.sin(2 * np.pi * np.arange(n)/7)

# External regressors
marketing_spend = np.random.normal(20, 5, n)
economic_index = np.random.normal(100, 10, n)

# Target variable
y = trend + yearly + weekly + 0.5*marketing_spend - 0.3*economic_index + np.random.normal(0, 3, n)

# Create DataFrame
df = pd.DataFrame({
    'ds': days,
    'y': y,
    'marketing_spend': marketing_spend,
    'economic_index': economic_index
})

print(f"Dataset Shape: {df.shape}")
print(df.head())

# -----------------------------
# Step 3: Initialize Prophet Model
# -----------------------------
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Add external regressors
model.add_regressor('marketing_spend')
model.add_regressor('economic_index')

# -----------------------------
# Step 4: Fit the Model
# -----------------------------
model.fit(df)

# -----------------------------
# Step 5: Forecast Future
# -----------------------------
future = model.make_future_dataframe(periods=90)  # next 90 days
future['marketing_spend'] = np.random.normal(20,5, len(future))
future['economic_index'] = np.random.normal(100,10, len(future))

forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Prophet Forecast with External Regressors")
plt.show()

# -----------------------------
# Step 6: Model Performance on Training Data
# -----------------------------
mae = mean_absolute_error(df['y'], forecast['yhat'][:len(df)])
rmse = np.sqrt(mean_squared_error(df['y'], forecast['yhat'][:len(df)]))

print("\n--- Forecast Performance ---")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# -----------------------------
# Step 7: Cross-Validation (Time Series)
# -----------------------------
# Adjust initial/horizon for small dataset
df_cv = cross_validation(model, initial='500 days', period='90 days', horizon='180 days')
df_p = performance_metrics(df_cv)

print("\n--- Cross-Validation Metrics ---")
print(df_p[['horizon','rmse','mae','mape']])

# -----------------------------
# Step 8: Feature Importance using SHAP
# -----------------------------
# Prophet is an additive model: trend + seasonality + regressors
# We focus on regressor importance using permutation approach
X = df[['marketing_spend','economic_index']]
y_target = df['y']

# Fit a simple XGBoost to get SHAP values for feature importance
xgb_model = XGBRegressor()
xgb_model.fit(X, y_target)

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X)

print("\n--- SHAP Feature Importance ---")
shap.summary_plot(shap_values, X, plot_type="bar")

# -----------------------------
# Step 9: Analyze Component Contributions
# -----------------------------
model.plot_components(forecast)
plt.show()

# -----------------------------
# Step 10: Sample Future Forecast with Explainability
# -----------------------------
sample = future.tail(5)
sample_forecast = model.predict(sample)
print("\n--- Sample Forecast for Next 5 Days ---")
print(sample_forecast[['ds','yhat','yhat_lower','yhat_upper']])

# Optional: SHAP values for regressors on future predictions
sample_X = sample[['marketing_spend','economic_index']]
sample_shap_values = explainer(sample_X)
shap.waterfall_plot(sample_shap_values[0])

print("\n--- Done ---")
