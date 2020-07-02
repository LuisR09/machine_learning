# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:11:11 2020

@author: USUARIO
"""

# Clasificación del usuario que visita un sitio web usa sistema operativo Windows,
#                                                               Macintosh o Linux
# Con Regresión Logística

# Importar librerías
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Cargar los datos
dataframe = pd.read_csv("usuarios_win_mac_lin.csv")
# duracion: duracion de la visita en segundos
# paginas: cantidad de paginas vistas durante la sesión
# acciones: cantidad de acciones del usuario(click, scroll, checkbox, sliders)
# valor: suma del valor de las acciones(cada accion lleva asociada una valoración de importancia)
# clase 0-Windows  1-Macintosh  2-Linux

dataframe.describe()

# cuantos registros hay por categoría
print(dataframe.groupby("clase").size())

# Visualización de datos en un histograma
dataframe.drop(["clase"],1).hist()
plt.show()

# Graficas de agrupamiento y relacion de las categorias
sb.pairplot(dataframe.dropna(), hue="clase", size=2, vars=["duracion","paginas","acciones","valor"],kind="reg")

# Definimos las variables de entrada(4) y salida
X = np.array(dataframe.drop(["clase"],1))
y = np.array(dataframe["clase"])
X.shape

model = linear_model.LogisticRegression(solver = "lbfgs")
model.fit(X,y)

predictions = model.predict(X)
print(predictions)

model.score(X,y)

# Dividir el conjunto en entrenamiento y prueba
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,y,test_size=0.2, random_state=seed)

name="Logistic Regression"
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_test)
print(accuracy_score(Y_test, predictions))

print(confusion_matrix(Y_test, predictions))

print(classification_report(Y_test, predictions))

X_new = pd.DataFrame({"duracion":[10], "paginas":[3], "acciones":[5], "valor":[9]})
model.predict(X_new)


