# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:15:41 2020

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("AAPL.csv")

X = dataset.iloc[:, 0].values.reshape(-1,1)
y = dataset.iloc[:, 4].values

X = np.array(range(254)).reshape(-1,1)

from sklearn.svm import SVR
svr_lin = SVR(kernel="linear", C=1e3).fit(X,y)
svr_poly = SVR(kernel = "poly", C=1e3, degree=2).fit(X,y)
svr_rbf = SVR(kernel = "rbf", C=1e3, gamma=0.1).fit(X,y)

y_pred_lin = svr_lin.predict(X)
y_pred_poly = svr_poly.predict(X)
y_pred_rbf = svr_rbf.predict(X)

svr_lin.score(X, y)
svr_poly.score(X, y)
svr_rbf.score(X, y)

plt.scatter(X, y, color="black", label="Data")
plt.plot(X, y_pred_lin, color="green", label="Linear model")
plt.plot(X, y_pred_poly, color="blue", label="Polynomial model")
plt.plot(X, y_pred_rbf, color="red", label="RBF model")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Support Vector Regression")
plt.legend()
plt.plot()

