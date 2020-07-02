# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:55:32 2020

@author: USUARIO
"""

# Agrupar usuarios Twitter de acuerdo a su personalidad
# Con K means - aprendizaje no supervisado 

# Importar las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from mpl_toolkits.mplot3d import Axes3D

# Cargar los datos
dataframe = pd.read_csv("analisis.csv")

dataframe.describe()

# cuantos registros hay por categoría
print(dataframe.groupby("categoria").size())
# categorias 1.actor/actriz 2.cantante 3.modelo 4.Tv,series 5.Radio 6.Tecnologia 7.Deportes 8.Politica 9.Escritor

# Visualización de datos en un histograma
dataframe.drop(["categoria"],1).hist()
plt.show()

# Graficas de agrupamiento y relacion de las categorias
sb.pairplot(dataframe.dropna(), hue='categoria',size=4,vars=["op","ex","ag"],kind='scatter') # No existe algun tipo de correlación o agrupamiento entre las categorias

# Definimos las variables de entrada(3) y salida
X = np.array(dataframe[["op","ex","ag"]])
y = np.array(dataframe["categoria"])

# Grafica en 3D con los 9 colores representando las categorías
fig = plt.figure()
ax = Axes3D(fig)
colores = ["NONE","red","green","blue","cyan","yellow","orange","black","pink","purple"] #El primer valor no lo toma
asignar=[]
for row in y:
    asignar.append(colores[row])
ax.scatter(X[:,0], X[:,1], X[:,2], c=asignar, s=60)

# Método del codo
wcss = []
for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,21), wcss)
plt.title("Elbow Curve")
plt.xlabel("Number of clusters")
plt.ylabel("wcss(k)")
plt.show()

# Ejecutamos el algoritmo
kmeans = KMeans(n_clusters=5).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)


# Gráficar en 3D los grupos y sus centroides
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
colores = ["red","green","blue","cyan","yellow"]
asignar = []
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0], X[:,1], X[:,2], c=asignar, s=60)
ax.scatter(C[:,0], C[:,1], C[:,2], marker="*", c=colores, s=1000)


# Graficar 2D "op" y "ex"
f1 = dataframe["op"].values
f2 = dataframe["ex"].values
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker="*", c=colores, s=1000)
plt.show()

# Graficar 2D "op" y "ag"
f1 = dataframe["op"].values
f2 = dataframe["ag"].values
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 2], marker="*", c=colores, s=1000)
plt.show()

# Graficar 2D "op" y "ag"
f1 = dataframe["ex"].values
f2 = dataframe["ag"].values
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 1], C[:, 2], marker="*", c=colores, s=1000)
plt.show()


# Cantidad de usuarios por clusters
copy = pd.DataFrame()
copy["usuario"]=dataframe["usuario"].values
copy["categoria"]=dataframe["categoria"].values
copy["label"] = labels
cantidadGrupo = pd.DataFrame()
cantidadGrupo["color"] = colores
cantidadGrupo["cantidad"] = copy.groupby("label").size()


# Cantidad y categoria en el primer cluster(0), en este caso color rojo
group_referrer_index = copy["label"] ==0
group_referrals = copy[group_referrer_index]
diversidadGrupo = pd.DataFrame()
diversidadGrupo["categoria"] = [0,1,2,3,4,5,6,7,8,9]
diversidadGrupo["cantidad"] = group_referrals.groupby("categoria").size()

# Usuarios que están mas cerca a los centroides de cada grupo
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,X)

users=dataframe["usuario"].values
for row in closest:
    print(users[row])


# Clasificar nuevas muestras
X_new = np.array([[45.92,57.74,15.66]]) # David Guetta

new_labels = kmeans.predict(X_new)
print(new_labels)





