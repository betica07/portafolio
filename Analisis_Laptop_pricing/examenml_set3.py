# -*- coding: utf-8 -*-
"""ExamenML_set3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BSeNTSOEiKbMvyAv9tTcAvMesQwXo7PS
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

os.chdir(r"/content/sample_data")

data = pd.read_csv("laptop_price.csv", encoding='latin-1')

display(data)

data.duplicated().sum()

data.isnull().sum()

num_cols = data.select_dtypes("number").columns
cat_cols = data.select_dtypes("object").columns

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

display(data)

data.duplicated().sum()

data = data.drop_duplicates()
data

import matplotlib.pyplot as plt
import seaborn as sns

# Create the boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='CPU_Company', y='Price (Euro)', data=data)
plt.title('Distribution of Laptop Prices by CPU Company')
plt.xlabel('CPU Company')
plt.ylabel('Price (Euro)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='TypeName', y='Price (Euro)', data=data)
plt.title('Distribution of Laptop Prices by TypeName')
plt.xlabel('TypeName')
plt.ylabel('Price (Euro)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Company', y='Price (Euro)', data=data)
plt.title('Distribution of Laptop Prices by Company')
plt.xlabel('Company')
plt.ylabel('Price (Euro)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

top_10_highest_price = data.sort_values(by=['Price (Euro)'], ascending=False).head(10)

display(top_10_highest_price)

plt.figure(figsize=(10, 6))
sns.histplot(data['Price (Euro)'], bins=30, kde=True)
plt.title('Distribution of Laptop Prices')
plt.xlabel('Price (Euros)')
plt.ylabel('Frequency')
plt.show()

for col in data.columns:
    unique_count = data[col].nunique()
    print(f"Column '{col}' has {unique_count} unique values.")

x = data.corr(numeric_only=True)

sns.heatmap(x, annot=True, cmap='crest')

plt.show()

"""Las caracteristicas que presentan una mayor correlatividad con el precio son la Ram y la frecuencia del CPU"""

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

display(data)

data.describe()

"""#Sin normalizar"""

X = data.drop(columns=['Price (Euro)'])
y = data['Price (Euro)']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#Regresion Lineal"""

# Regresion lineal
from sklearn import linear_model
regr = linear_model.LinearRegression()

# Entrenamos y evaluamos con los datos de train

regr.fit(X_train, y_train)
regr.score(X_train,y_train)

prediccion = regr.predict(X_test)
regr.score(X_test,y_test)

"""Estos resultados indican que el modelo no esta sobreentrenado pero no es lo suficientemente bueno"""

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

modelo=sm.OLS(y_train, X_train)
result=modelo.fit()
result.summary()

# Con Constante
train_data_X = sm.add_constant(X_train)
modelo=sm.OLS(y_train, train_data_X)
result=modelo.fit()
result.summary()

"""#DecisionTree"""

from sklearn.tree import DecisionTreeRegressor

# Definimos el modelo y el numero maximo de ramas del arbol.

arbol = DecisionTreeRegressor(criterion='squared_error', max_depth=8, max_features=None, max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

# Entrenamos y evaluamos con los datos de train

arbol=arbol.fit(X_train, y_train)
arbol.score(X_train, y_train)
# Realizamos la prediccion
# Evaluamos en los datos de test.

prediccionarbol = arbol.predict(X_test)
arbol.score(X_test, y_test)

"""Esta diferencia en las metricas indica que el modelo esta sobreentrenado"""

importancias=pd.DataFrame(arbol.feature_importances_)
importancias.index=(X.columns)
importancias.sort_values(by=0, ascending=False)

"""#RandomForest"""

from sklearn.ensemble import RandomForestRegressor

RF= RandomForestRegressor(n_estimators=500, criterion='squared_error' ,max_features='sqrt' ,max_depth=30, min_samples_split=2, min_samples_leaf=3, max_leaf_nodes=None,min_impurity_decrease=0, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)

clf = RF.fit(X_train,y_train)
# Analizamos la fiabilidad sobre los datos utilizados para crear el modelo.

RF.score(X_train,y_train)

y_test_pred = RF.predict(X_test)
test_score = r2_score(y_test, y_test_pred)
print(f"Testing R² Score: {test_score}")

# El ultimo paso es determinar la importancia de cada una de las variables

importancias=pd.DataFrame(clf.feature_importances_)
importancias.index=(X.columns)
importancias.sort_values(by=0, ascending=False)

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import numpy as np

# Fit the model using training data
clf.fit(X_train, y_train)

# Predict values for both training and testing sets
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Metrics for Training Set
print("Training Set Metrics:")
print("R² Score (Train):", r2_score(y_train, y_train_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_train, y_train_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_train, y_train_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Explained Variance Score:", explained_variance_score(y_train, y_train_pred))

# Metrics for Testing Set
print("\nTesting Set Metrics:")
print("R² Score (Test):", r2_score(y_test, y_test_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_test_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Explained Variance Score:", explained_variance_score(y_test, y_test_pred))

"""Los modelos de RandomForest y XGBoost ofrecen los mejores resultados, además de significativas similitudes en la importancia de las caracteristicas, puesto que las dos primeras coinciden"""

import matplotlib.pyplot as plt
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

"""#RandomForest sin los 10 precios extremos"""

# Create a copy of the DataFrame
data_no_top10 = data.copy()

# Remove the top 10 highest price rows
data_no_top10 = data_no_top10.drop(top_10_highest_price.index)

# Display the new DataFrame (optional)
display(data_no_top10)

Xr = data_no_top10.drop(columns=['Price (Euro)'])
yr = data_no_top10['Price (Euro)']

# Split into training and testing sets
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

RF= RandomForestRegressor(n_estimators=500, criterion='squared_error' ,max_features='sqrt' ,max_depth=30, min_samples_split=2, min_samples_leaf=3, max_leaf_nodes=None,min_impurity_decrease=0, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)

clf = RF.fit(Xr_train,yr_train)
# Analizamos la fiabilidad sobre los datos utilizados para crear el modelo.

RF.score(Xr_train,yr_train)

yr_test_pred = RF.predict(Xr_test)
test_score = r2_score(yr_test, yr_test_pred)
print(f"Testing R² Score: {test_score}")

importancias=pd.DataFrame(clf.feature_importances_)
importancias.index=(X.columns)
importancias.sort_values(by=0, ascending=False)

"""Al eliminar 10 valores extremos aumenta la R2 del training y aumenta ligeramente la R2 del testing, por tanto los valores extremos no afectan significativamente la capacidad predictiva del modelo

#Probando otro encoding
"""

from sklearn.preprocessing import OneHotEncoder

# Create a OneHotEncoder object
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical features
categorical_cols = ['Company', 'Product', 'TypeName', 'ScreenResolution', 'CPU_Company', 'CPU_Type', 'Memory', 'GPU_Company', 'GPU_Type', 'OpSys']
encoded_features = encoder.fit_transform(data[categorical_cols])

# Create a new DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# Drop the original categorical columns from the data DataFrame
data = data.drop(columns=categorical_cols)

# Concatenate the encoded DataFrame with the original DataFrame and name it differently
data_other_encoding = pd.concat([data, encoded_df], axis=1)

display(data_other_encoding)

X = data_other_encoding.drop(columns=['Price (Euro)'])
y = data_other_encoding['Price (Euro)']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#Regresion Lineal"""

# Regresion lineal
from sklearn import linear_model
regr = linear_model.LinearRegression()

# Entrenamos y evaluamos con los datos de train

regr.fit(X_train, y_train)
regr.score(X_train,y_train)

prediccion = regr.predict(X_test)
regr.score(X_test,y_test)

"""#DecisionTree"""

from sklearn.tree import DecisionTreeRegressor

# Definimos el modelo y el numero maximo de ramas del arbol.

arbol = DecisionTreeRegressor(criterion='squared_error', max_depth=8, max_features=None, max_leaf_nodes=None,min_impurity_decrease=0.0,min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

# Entrenamos y evaluamos con los datos de train

arbol=arbol.fit(X_train, y_train)
arbol.score(X_train, y_train)
# Realizamos la prediccion
# Evaluamos en los datos de test.

prediccionarbol = arbol.predict(X_test)
arbol.score(X_test, y_test)

"""#RandomForest"""

from sklearn.ensemble import RandomForestRegressor

RF= RandomForestRegressor(n_estimators=500, criterion='squared_error' ,max_features='sqrt' ,max_depth=300, min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=None,min_impurity_decrease=0, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)

clf = RF.fit(X,y)
# Analizamos la fiabilidad sobre los datos utilizados para crear el modelo.

RF.score(X,y)

# Procedemos a realizar validacion cruzada

from sklearn.model_selection import cross_val_score


scores = cross_val_score(RF, X, y, cv=10)

print(scores.mean())

"""El OneHotEncoder altera significativamente la dimensionalidad del dataset añadiendo riesgo de sobreentrenamiento.

En el modelo de regresion lineal, el R2 de training indica sobreentrenamiento dada la gran cantidad de características y de acuerdo al R2 de test (0.8188) generaliza mejor respecto a la codificacion anterior.

En el modelo DecisionTree se observa una mejora en la R2 de test, que sugiere que generaliza mejor dado el mayor numero de características.

En el modelo RandomForest, dada sus metricas se puede concluir que la alteración en la dimensionalidad del dataset introduce "ruido", y reduce significativamente la capacidad de generalizacion del modelo
"""