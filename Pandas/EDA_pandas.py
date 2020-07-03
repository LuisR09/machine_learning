# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:13:47 2020

@author: USUARIO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

url = 'https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv'

df = pd.read_csv(url, sep=";")

print("Cantidad de filas y columnas:", df.shape)
print("Nombre columnas:", df.columns)

# Columnas, nulos y tipos de datos
df.info()

# Descripción estadística de los datos numéricos
df.describe()


# correlación entre los datos
corr = df.set_index("alpha_3").corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()
# se observa baja correlación entre las variables


# Cargamos un segundo archivo para observar el crecimiento de la población en españa
url = 'https://raw.githubusercontent.com/DrueStaples/Population_Growth/master/countries.csv'
df_pop = pd.read_csv(url)
print(df_pop.head())

df_pop_es = df_pop[df_pop["country"] == 'Spain' ]
print(df_pop_es.head())
df_pop_es.drop(['country'],axis=1)['population'].plot(kind='bar')


# Hacemos una comparación con otro pais
df_pop_ar = df_pop[(df_pop["country"] == 'Argentina')]

anios = df_pop_es['year'].unique()

pop_ar = df_pop_ar['population'].values
pop_es = df_pop_es['population'].values

# creamos un dataframe nuevo
df_plot = pd.DataFrame({'Argentina': pop_ar,
                    'Spain': pop_es}, 
                       index=anios)
df_plot.plot(kind='bar')

# Ahora filtrar todos los paises hispano-hablantes
df_espanol = df.replace(np.nan, "", regex=True)
df_espanol = df_espanol[df_espanol["languages"].str.contains("es")]


# Visualizamos
df_espanol.set_index("alpha_3")[["population", "area"]].plot(kind="bar",rot=65,figsize=(20,10))


# Detectar outliers
anomalies = []

# Función ejemplo para detección de outliers
def find_anomalies(data):
    # Set upper and lower limit to 2 standard deviation
    data_std = data.std()
    data_mean = data.mean()
    anomaly_cut_off = data_std * 2
    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    print(lower_limit.iloc[0])
    print(upper_limit.iloc[0])
    
    # Generate outliers
    for index, row in data.iterrows():
        outlier = row # # obtener primer columna
        # print(outlier)
        if (outlier.iloc[0] > upper_limit.iloc[0]) or (outlier.iloc[0] < lower_limit.iloc[0]):
            anomalies.append(index)
    return anomalies

find_anomalies(df_espanol.set_index('alpha_3')[['population']])


# Quitamos BRA y USA por ser outlier y volvamos a graficar
df_espanol.drop([30, 233], inplace=True)
df_espanol.set_index("alpha_3")[["population", "area"]].sort_values(["population"]).plot(kind="bar", rot=65, figsize=(16,9))






