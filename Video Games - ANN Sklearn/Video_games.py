# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:14:08 2020

@author: USUARIO
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


juegos = pd.read_csv("vgsales.csv")

juegos.info()
pd.isnull(juegos).sum()

juegos = juegos.replace(np.nan, "0")

juegos["Platform"] = juegos["Platform"].replace("2600", "Atari")

juegos["Year"].unique()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

juegos["plataforma"] = encoder.fit_transform(juegos.Platform)
juegos["publica"] = encoder.fit_transform(juegos.Publisher)

print(juegos.plataforma.unique())

juegos.columns

X = juegos[['plataforma', 'publica', 'Global_Sales', ]]
y = juegos['Genre']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, alpha=0.0001,
                    solver='adam', random_state=21, tol=0.000000001)
#mlp = MLPClassifier(hidden_layer_sizes=(6,6,6,6), max_iter=5000)

mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

pred2 = mlp.predict([[15.6, 3.7, 35.5]])


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

