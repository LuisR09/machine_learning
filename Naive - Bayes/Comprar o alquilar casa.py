# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:39:05 2020

@author: USUARIO
"""

# Comprar o alquilar casa
# Con Naive - Bayes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest

dataframe = pd.read_csv("comprar_alquilar.csv")
dataframe.head(10)

dataframe["comprar"].value_counts() #contar un valor de una columna
print(dataframe.groupby("comprar").size()) #contar un valor de una columna

# Mirar la data en forma de histograma
dataframe.drop(["comprar"], axis=1).hist()
plt.show()

# Agregar columnas nuevas y eliminar las menos relevantes
dataframe["gastos"] = (dataframe["gastos_comunes"]+dataframe["gastos_otros"]+dataframe["pago_coche"])
dataframe["financiar"] = (dataframe["vivienda"]-dataframe["ahorros"])
dataframe.drop(["gastos_comunes","gastos_otros","pago_coche"], axis=1)

reduced = dataframe.drop(["gastos_comunes","gastos_otros","pago_coche"], axis=1)
reduced.describe()

# Features Selection. Seleccionar las mejores caracteristicas, en este caso es 5
X = dataframe.drop(["comprar"], axis=1)
y = dataframe["comprar"]

best=SelectKBest(k=5)
X_new = best.fit_transform(X, y)
X_new.shape
selected = best.get_support(indices=True)
print(X.columns[selected])

used_features = X.columns[selected]

# Graficar la correlación de las 5 características seleccionadas
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
sb.heatmap(dataframe[used_features].astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True)

# Dividir la data en entrenamiento y prueba
X_train, X_test = train_test_split(dataframe, test_size=0.2, random_state=6)
y_train = X_train["comprar"]
y_test = X_test["comprar"]

# Uso del algoritmo
gnb = GaussianNB()
gnb.fit(X_train[used_features].values, y_train)

y_pred = gnb.predict(X_test[used_features])

print("Precisión en el set de Entrenamiento:{:.4f}".format(gnb.score(X_train[used_features], y_train)))
print("Precisión en el set de Test:{:.4f}".format(gnb.score(X_test[used_features], y_test)))

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

#                   ["Ingresos", "Ahorros", "Hijos", "Trabajo", "Financiar"]
print(gnb.predict([[    2000,       5000,       0,        5,        200000],
                  [     6000,       34000,      2,        5,        320000]]))
    


# Los resultados indican que el primero debe alquilar[0] y el segundo debe comprar[1]

