# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:42:03 2024

@author: rportatil108
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import r2_score



os.chdir(r"C:\Users\rportatil108\Documents\Beatriz\02_Modulo_Machine_Learning\examen")

data = pd.read_csv("laptop_price.csv", encoding='latin-1')



#################

import pandas as pd

# Define a function to extract the CPU information
def extract_cpu_info(cpu_string):
    parts_cpu = cpu_string.split()
    company_cpu = parts_cpu[0]  # First part is the company
    type_cpu = ' '.join(parts_cpu[1:-1])  # Everything after the first part except the last
    frequency = parts_cpu[-1].replace('GHz', '')  # Get the last part and remove 'GHz'

    # Convert frequency to float if possible
    try:
        frequency = float(frequency)
    except ValueError:
        frequency = None  # Handle cases where conversion fails

    return pd.Series([company_cpu, type_cpu, frequency])


# Define a function to extract the GPU information
def extract_GPU_info(GPU_string):
    parts_GPU = GPU_string.split()
    company_GPU = parts_GPU[0]
    type_GPU = ' '.join(parts_GPU[1:])
    return pd.Series([company_GPU, type_GPU])


# Apply the function to the 'Cpu' & 'Gpu' column and create new columns:
data[['CPU_Company', 'CPU_Type', 'CPU_Frequency (GHz)']] = data['Cpu'].apply(extract_cpu_info)
data[['GPU_Company', 'GPU_Type']] = data['Gpu'].apply(extract_GPU_info)


# Clean the Weight column
data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)
data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)


# Drop columns 'Cpu' & 'Gpu', because new columns were created for them:
data.drop(columns=['Cpu', 'Gpu', 'laptop_ID'], inplace=True)


# Reorder columns to organize:
new_order = ['Company', 'Product', 'TypeName', 'Inches', 'ScreenResolution',
             'CPU_Company', 'CPU_Type', 'CPU_Frequency (GHz)', 'Ram', 'Memory',
             'GPU_Company', 'GPU_Type', 'OpSys', 'Weight', 'Price_euros']
data = data[new_order]


# Rename 'Ram' & 'Weight' & 'Price' column:
data.rename(columns={'Ram': 'RAM (GB)' , 'Weight': 'Weight (kg)' , 'Price_euros':'Price (Euro)'}, inplace=True)


###################
data = data.drop_duplicates()


encoder = LabelEncoder()

data['Company'] = encoder.fit_transform(data['Company'])
data['Product'] = encoder.fit_transform(data['Product'])
data['TypeName'] = encoder.fit_transform(data['TypeName'])
data['ScreenResolution'] = encoder.fit_transform(data['ScreenResolution'])
data['CPU_Company'] = encoder.fit_transform(data['CPU_Company'])
data['CPU_Type'] = encoder.fit_transform(data['CPU_Type'])
data['Memory'] = encoder.fit_transform(data['Memory'])
data['GPU_Company'] = encoder.fit_transform(data['GPU_Company'])
data['GPU_Type'] = encoder.fit_transform(data['GPU_Type'])
data['OpSys'] = encoder.fit_transform(data['OpSys'])




X = data.drop(columns=['Price (Euro)'])
y = data['Price (Euro)']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(
    base_score=1, colsample_bylevel=0.8, colsample_bytree=0.7, gamma=0.5, 
    learning_rate=0.1, max_delta_step=0, max_depth=5, min_child_weight=5, 
    n_estimators=200, objective='reg:squarederror', reg_alpha=0, reg_lambda=1, 
    scale_pos_weight=1, seed=0, subsample=0.8
)



# Train the model on the training set
modeloxgb = xgb_model.fit(X_train, y_train)

# Evaluate the model on the training set
train_score = xgb_model.score(X_train, y_train)
print(f"Training R² Score: {train_score}")

# Evaluate the model on the testing set
y_test_pred = xgb_model.predict(X_test)
test_score = r2_score(y_test, y_test_pred)
print(f"Testing R² Score: {test_score}")

# Perform cross-validation on the training set
scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validated R² Score (Training Set): {scores.mean()}")

# Plot feature importance
plt.figure(figsize=(10, 8))  # Adjust the figure size if necessary
plot_importance(modeloxgb, importance_type='gain', xlabel='Feature Importance')
plt.title('Feature Importance')
plt.show()

# Create a DataFrame for feature importance
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': modeloxgb.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display the sorted DataFrame
print(importances)


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Define the model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    seed=0
)

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    scoring='r2',  # R² as the scoring metric
    cv=5,
    n_iter=50,      # Number of random combinations
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit RandomizedSearchCV to the data
random_search.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation R²:", random_search.best_score_)

import warnings
warnings.filterwarnings("ignore")



from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from xgboost import plot_importance
import matplotlib.pyplot as plt

# Predict on the training and testing datasets
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Calculate Metrics for Training Set
print("Training Set Metrics:")
print("R² Score (Train):", r2_score(y_train, y_train_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_train, y_train_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_train, y_train_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Explained Variance Score:", explained_variance_score(y_train, y_train_pred))
print("Mean Absolute Percentage Error (MAPE):", np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100, "%")

# Calculate Metrics for Testing Set
print("\nTesting Set Metrics:")
print("R² Score (Test):", r2_score(y_test, y_test_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_test_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Explained Variance Score:", explained_variance_score(y_test, y_test_pred))
print("Mean Absolute Percentage Error (MAPE):", np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100, "%")

# Cross-Validated Metrics
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
cv_r2 = cross_val_score(xgb_model, X_train, y_train, scoring='r2', cv=5)
cv_mae = cross_val_score(xgb_model, X_train, y_train, scoring=mae_scorer, cv=5)
cv_mse = cross_val_score(xgb_model, X_train, y_train, scoring=mse_scorer, cv=5)

print("\nCross-Validated Metrics on Training Set:")
print("Cross-Validated R²:", cv_r2.mean())
print("Cross-Validated MAE:", -cv_mae.mean())
print("Cross-Validated MSE:", -cv_mse.mean())


######################### Normalizando ################################3

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame(data)


features = df.drop(columns=['Price (Euro)'])
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
normalized_df['Price (Euro)'] = df['Price (Euro)']  # Retain original target values

# Display the standardized DataFrame
print("Original Data:")
print(df)
print("\nZ-Score Normalized Data:")
print(normalized_df)


Xn = normalized_df.drop(columns=['Price (Euro)'])
yn = normalized_df['Price (Euro)']

# Split into training and testing sets
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.2, random_state=42)



datosN = xgb.DMatrix(Xn, label=yn)

xgb_model = xgb.XGBRegressor(base_score=1, colsample_bylevel=0.8, colsample_bytree=0.7, gamma=0.5, learning_rate=0.1, max_delta_step=0, max_depth=5, min_child_weight=5, missing=1, n_estimators=200, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=0.8)
xgb_model

modeloxgb = xgb_model.fit(Xn,yn)
modeloxgb

xgb_model.score(Xn,yn)
scores = cross_val_score(xgb_model, Xn, yn, cv=5)
scores.mean()

## La normalizacion no aporta diferencias significativas a las metricas pues los modelos como Random Forest o XGBoost no se "fijan" en las escalas de las caracteristicas
