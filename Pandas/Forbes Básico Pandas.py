# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:18:58 2020

@author: USUARIO
"""

import pandas as pd

data = pd.read_csv("Forbes.csv")

# Lista las 6 primeras filas
print(data[:6])

# Lista la primera fila
print(data.loc[0])

# Lista sólo 2 columnas
print(data[["Company", "Sales"]])

# Suma de la columna Sales
print(sum(data["Sales"]))

# Promedio de las columnas numéricas
print(data.mean())

# Lista valores de Sales menores a 0.5
print(data[data["Sales"]<0.5])





