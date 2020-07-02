# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:31:50 2020

@author: USUARIO
"""

# Opiniones de usuarios sobre una app
# Usamos KNN


# Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Importar el dataset
dataframe = pd.read_csv("reviews_sentiment.csv", sep=";")

dataframe.describe()

dataframe.hist()

print(dataframe.groupby("Star Rating").size())


sb.catplot("Star Rating", data=dataframe, kind="count", aspect=3)
sb.catplot("wordcount", data=dataframe, kind="count", aspect=3)


# Preparamos las entradas
X = dataframe[["wordcount","sentimentValue"]].values
y = dataframe["Star Rating"].values

# Escalar las variables de entrada
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Hallar el numero optimo de K
k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.scatter(k_range, scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.xticks([0,5,10,15,20])
plt.show()
# El resultado óptimo es 7


# Uso del algoritmo
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print("Accuracy of K-NN classifier on training set: {:.4f}"
      .format(knn.score(X_train, y_train)))
print("Accuracy of K-NN classifier on test set: {:.2f}"
      .format(knn.score(X_test, y_test)))


pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

from sklearn.metrics import accuracy_score
exactitud = accuracy_score(y_test, pred)


# Grafica
h = .02  # step size in the mesh
 
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff','#c2f0c2'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933','#FFFF00','#00ffff','#00FF00'])
 
# we create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)
 
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
    
patch0 = mpatches.Patch(color='#FF0000', label='1')
patch1 = mpatches.Patch(color='#ff9933', label='2')
patch2 = mpatches.Patch(color='#FFFF00', label='3')
patch3 = mpatches.Patch(color='#00ffff', label='4')
patch4 = mpatches.Patch(color='#00FF00', label='5')
plt.legend(handles=[patch0, patch1, patch2, patch3,patch4])
 
plt.title("5-Class classification (k = %i)" % n_neighbors)
 
plt.show()



# Nuevos datos
print(clf.predict([[5, 1.0]]))  # (numero de palabras, sentimiento de valor). Lo clasifica como [5]
print(clf.predict_proba([[5, 1.0]])) # clasifica la probabilidad de cada valor de estrella



