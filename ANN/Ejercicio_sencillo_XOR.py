# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:03:33 2020

@author: USUARIO
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense


# Cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]])

# Y estos son los resultados que se obtienen en el mismo orden
target_data = np.array([[0],[1],[1],[0]])

# Creamos la arquitectura de la red neuronal
model = Sequential()
model.add(Dense(16, input_dim=2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Ajustes del modelo
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["binary_accuracy"])

# Entrenar la red
model.fit(training_data, target_data, epochs=1000) 

# Evaluar el modelo
scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predicción del modelo
print(model.predict(training_data).round())

