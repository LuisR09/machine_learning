# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:55:07 2020

@author: USUARIO
"""

# Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Cargamos los datos de entrada
data = pd.read_csv("articulos_ml.csv")

data.describe()

# visualizamos rapidamente las caracteríasticas de entrada
data.drop(["Title","url","Elapsed days"],1).hist()
plt.show()

# Vamos a recotar los datos en la zona donde se concentran mas lo puntos
# Esto es en el eje X: entre 0 y 3.500
# y en el eje Y: entre 0 y 80.000
filtered_data = data[(data["Word count"] <= 3500) & (data["# Shares"] <= 80000)]


# Asignamos nuestra variables de entrada X para entrenamiento y las etiquetas Y
dataX = filtered_data[["Word count"]]
X_train = np.array(dataX)                   
y_train = filtered_data["# Shares"].values
                   
# creamos el objeto de regresión linear
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
              
# se calcula la predicción
y_pred = regr.predict(X_train)

# calcular el coeficiente
print("El coeficiente es:", regr.coef_)
# calcular el intercepto con el eje y
print("El intercepto es:", regr.intercept_)     
                 
# El error cuadrado medio
print("El error cuadrado medio es:", mean_squared_error(y_train, y_pred))                   

# Puntaje del r2, hay dos formas de hacerlo
print("r2: %.4f" % regr.score(X_train, y_train))
print("r2: %.4f" % r2_score(y_train, y_pred))                 
                   
# Graficamos los datos junto con el modelo                  
plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred, color="blue")
plt.title("Regresión lineal")
plt.xlabel("Cantidad de palabras")
plt.ylabel("Compartido en redes")
plt.show()               
                   
# Ejemplo comprobando cuanto se va a compartir dandole un dato de entrada
# Quiero predecir cuantos "shares" voy a obtener por un artículo con 2.000 palabras
y_dosmil = regr.predict([[2000]])           
                   

# Vamos a mejorar el modelo con una dimensión más:
# Para poder graficar en 3D, haremos una vble nueva que será la suma de los enlaces, comentarios e imágenes                   

suma = filtered_data["# of Links"] + filtered_data["# of comments"].fillna(0) + filtered_data["# Images video"]

dataX2 = pd.DataFrame()
dataX2["Word count"] = filtered_data["Word count"]
dataX2["suma"] = suma        
XY_train = np.array(dataX2)   
z_train = filtered_data["# Shares"].values                           
                      
# creamos un nuevo objeto de regresión lineal
regr2 = linear_model.LinearRegression()                     

regr2.fit(XY_train, z_train)

# Predicción
z_pred = regr2.predict(XY_train)
           
# Los coeficientes           
print("Coeficiente", regr2.coef_)                      
# Intercepto eje Z
print("intercepto", regr2.intercept_)

# Error cuadratico medio
print("Error cuadratico medio", mean_squared_error(z_train, z_pred))

# Puntaje del r2
print("r2: %.4f" % regr2.score(XY_train, z_train))
print("r2: %.4f" % r2_score(z_train, z_pred))


# Graficar en 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

# Creamos una malla, sobre la cual graficamos el plano
xx, yy = np.meshgrid(np.linspace(0, 3500, num=10), np.linspace(0, 65, num=10))

# Calculamos los valores del plano para los puntos x e y            
nuevoX = regr2.coef_[0] * xx
nuevoY = regr2.coef_[1] * yy

# Calculamos los correspondientes valores para Z. se debe sumar el punto de intercepción
z = nuevoX + nuevoY + regr2.intercept_

# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap="hot")

# Graficamos en azul los puntos de entrenamiento
ax.scatter(XY_train[:,0], XY_train[:,1], z_train, c="blue")
# Graficamos en rojo los puntos de predicción
ax.scatter(XY_train[:,0], XY_train[:,1], z_pred, c="red")
    
# Cambio la posición de la camara
ax.view_init(elev=30, azim=50)


# si quiero prededir cuantas cuantos "shares" voy a obtener por un articulo con:
# 2000 palabras y con enlaces: 10, comentarios:4, imagenes:6
z_dosmil = regr2.predict([[2000, 10+4+6]])
print(int(z_dosmil))

       