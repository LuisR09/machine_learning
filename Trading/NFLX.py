# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:04:02 2020

@author: USUARIO
"""

# Install the libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.style.use('bmh')

# Get the data
df = pd.read_csv('NFLX.csv')

# Visualize the close price data
plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.title('Netflix stock price')
plt.xlabel("Days")
plt.ylabel("Close Price(usd)")
plt.show()

# Number of backtesting
bt=25

# Get the close price
df = df[["Close"]]
df["Prediction"] = df.shift(-bt)

# Create the feature and target data set
X = df[["Close"]][:-bt]
y = df[["Prediction"]][:-bt]

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Create the models
# Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(X_train, y_train)
# Create the linear regression model
lr = LinearRegression().fit(X_train, y_train)

# Create the valid "x"
X_future = X.tail(bt)

# Show the model tree prediction
tree_prediction = tree.predict(X_future)

# Show the model Linear regression
lr_prediction = lr.predict(X_future)

y_true = y.tail(bt)

tree_prediction = np.array(tree_prediction)
lr_prediction = np.array(lr_prediction)
y_true = np.array(y_true)

rmse_tree = np.sqrt(np.mean((tree_prediction - y_true)**2))
rmse_lr = np.sqrt(np.mean((lr_prediction - y_true)**2))


# Visualize the data
predictions = tree_prediction

valid = df[X.shape[0]:][["Close"]]
valid["Predictions"] = predictions
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Days")
plt.ylabel("Close price(usd)")
plt.plot(df['Close'])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Origin", "val", "Predictions"])
plt.show()



predictions = lr_prediction

valid = df[X.shape[0]:][["Close"]]
valid["Predictions"] = predictions
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Days")
plt.ylabel("Close price(usd)")
plt.plot(df['Close'])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Origin", "val", "Predictions"])
plt.show()



