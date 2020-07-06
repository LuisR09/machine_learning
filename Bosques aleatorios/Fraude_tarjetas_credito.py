# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:48:16 2020

@author: USUARIO
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("creditcard.csv")

df["Class"].value_counts()

# Gráfica
count_classes = df["Class"].value_counts()
count_classes.plot(kind = "bar", rot=0)
LABELS = ["Normal", "Fraud"]
plt.xticks(range(2), LABELS)
plt.title("Frequency by observation number")
plt.xlabel("class")
plt.ylabel("Number of observations")

# Definimos nuestras etiquetas y features
X = df.drop("Class", 1)
y = df["Class"]

# Dividimos en sets de entramiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
# crear el modelo con 100 árboles
model = RandomForestClassifier(n_estimators=100,
                               bootstrap = True, verbose=2,
                               max_features = "sqrt")
# bootstrap: utiliza diversos tamaños de muestras para entrenar, si pone en falso utilizará siempre el dataset completo
model.fit(X_train, y_train)

pred_y = model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, pred_y)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("True Class")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, pred_y))


# a pesar de tener clases tan desbalanceadas, el modelo genera un buen recall para el "fraude"



