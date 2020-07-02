# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:55:01 2020

@author: USUARIO
"""

# Importar Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargamos los datos de entrada
dataframe = pd.read_csv("comprar_alquilar.csv")
print(dataframe.tail(10))

# Normalizamos los datos
scaler = StandardScaler()
df = dataframe.drop(["comprar"], axis=1) # Quito la vable idependiente "Y"
scaler.fit(df) # calcula la media para poder hacer la transformacion
X_scaled = scaler.transform(df) # convertimos nuestros datos con las nuevas dimensiones de PCA

# Instanciamos objeto PCA y aplicamos
pca = PCA(n_components=9) # Otra opción es instanciar pca sólo con dimensiones nuevas hasta obtener un mínimo "explicado" ej:pca=PCA(0.85)
pca.fit(X_scaled) # obtener los componentes principales
X_pca = pca.transform(X_scaled) # convertimos nuestros datos con las nuevas dimensiones de PCA

print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print("suma:", sum(expl[0:5])) # Vemos que con 5 componentes tenemos algo mas del 85% de varianza explicada

# graficamos el acumulado de varizna explicada en las nuevas dimensiones
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()

# graficamos en 2 dimensiones, tomando los 2 primeros componentes principales
Xax=X_pca[:,0]
Yax=X_pca[:,1]
labels=dataframe['comprar'].values
cdict={0:'red',1:'green'}
labl={0:'Alquilar',1:'Comprar'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix=np.where(labels==l)
    ax.scatter(Xax[ix],Yax[ix],c=cdict[l],label=labl[l],s=40,marker=marker[l],alpha=alpha[l])
 
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()






