Advanced Time Series Forecasting with Prophet + XAI
Project Description

This project demonstrates advanced time series forecasting using the Facebook Prophet library, incorporating external regressors and explainable AI (XAI) techniques. The goal is to forecast a time series with trends, multiple seasonalities, and external influences, while providing interpretability of the predictions.

The project includes:

Generation of a synthetic dataset with daily observations over 3 years.

Integration of external regressors (marketing_spend and economic_index) into the Prophet model.

Model fitting, forecasting, and evaluation using metrics like MAE and RMSE.

Time series cross-validation for model robustness.

Feature importance analysis using SHAP.

Component contribution analysis (trend, weekly and yearly seasonality, regressors).

Dataset

A programmatically generated dataset with the following columns:

Column	Description
ds	Date of observation
y	Target variable
marketing_spend	External regressor representing daily marketing spend
economic_index	External regressor representing economic conditions

Shape: (1096, 4) (3 years of daily data)

Example:

          ds          y  marketing_spend  economic_index
0 2020-01-01  -2.921191        22.48          100.79
1 2020-01-02  -4.322760        19.31           80.02
2 2020-01-03  -5.268728        23.24          109.16
3 2020-01-04  -6.422633        27.62          103.46
4 2020-01-05 -13.637028        18.83          109.98

Installation
Requirements

Python 3.9+

Libraries listed in requirements.txt:

pandas==2.1.1
numpy==1.26.2
matplotlib==3.8.0
prophet==1.2
scikit-learn==1.3.0
xgboost==1.7.6
shap==0.42.1


Install dependencies:

pip install -r requirements.txt


Note: Prophet requires cmdstanpy for model fitting. If any installation issues occur, run:

pip install cmdstanpy

Usage

Run the Python script:

python forecast_prophet_xai.py


Workflow in the script:

Generate synthetic dataset with trend, seasonality, and regressors.

Initialize Prophet with yearly and weekly seasonality.

Add external regressors (marketing_spend, economic_index).

Fit the Prophet model and generate forecasts for the next 90 days.

Evaluate model performance using MAE and RMSE.

Perform time series cross-validation using Prophet's cross_validation.

Analyze feature importance using SHAP and XGBoost.

Visualize forecast components (trend, seasonality, regressors).

Sample forecast with SHAP waterfall plots for explainability.

Results
Forecast Performance
MAE: 5.154
RMSE: 6.356

Cross-Validation Metrics (sample)
     horizon      rmse       mae      mape
0    18 days  2.897      2.330    1.667
1    19 days  2.820      2.248    1.661
...
180 180 days  3.193      2.652    0.126

Sample Forecast for Next 5 Days
          ds       yhat  yhat_lower  yhat_upper
0 2023-03-27  40.51   36.92   44.27
1 2023-03-28  36.55   32.75   40.25
2 2023-03-29  46.10   42.16   49.87
3 2023-03-30  45.35   41.53   48.99
4 2023-03-31  50.65   47.02   54.30

SHAP Feature Importance

marketing_spend and economic_index contributions analyzed using SHAP summary plots.

Shows the impact of each external regressor on forecast predictions.

Component Analysis

Trend, weekly, yearly seasonality, and regressors contributions are visualized using Prophet’s plot_components.

Notes

Dataset is synthetic for demonstration purposes but simulates real-world complexities like multiple seasonalities and external influences.

Hyperparameters (seasonality, changepoint prior) can be tuned further for production use.

The workflow demonstrates explainable AI (XAI) on time series models.
