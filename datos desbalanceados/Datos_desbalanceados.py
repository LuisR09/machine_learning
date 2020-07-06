# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:09:39 2020

@author: USUARIO
"""

pip install -U imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier

from collections import Counter


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
# Se observa una cantidad muy desbalanceada


# Definimos nuestras etiquetas y features
X = df.drop("Class", 1)
y = df["Class"]

# Dividimos en sets de entramiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Creamos una función que crea el modelo que usaremos cada vez
def run_model(X_train, X_test, y_train, y_test):
    clf_base = LogisticRegression(C=1.0, penalty="l2", random_state=1, solver="newton-cg")
    clf_base.fit(X_train, y_train)
    return clf_base

# Ejecutamos el modelo "tal cual"
model = run_model(X_train, X_test, y_train, y_test)

# Definimos una función para mostrar los resultados
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True Class")
    plt.show()
    print(classification_report(y_test, pred_y))

pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)


# Modelo balanceado
def run_model_balanced(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1.0,penalty="l2", random_state=1, solver="newton-cg", class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf

model = run_model_balanced(X_train, X_test, y_train, y_test)

pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)


# Modelo, estratégia subsampling en la clase mayoritaria (eliminar mucha de la clase mayoritaria)
us = NearMiss(ratio=0.5, n_neighbors=3, version=2)
X_train_res, y_train_res = us.fit_sample(X_train, y_train)

print("Distribution before resampling {}".format(Counter(y_train)))
print("Distribution after resampling {}".format(Counter(y_train_res)))

model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)


# Modelo, estratégia Oversampling en la clase minoritaria (aumentar la clase minoritaria)
os =  RandomOverSampler(ratio=0.5)
X_train_res, y_train_res = os.fit_sample(X_train, y_train)

print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution labels after resampling {}".format(Counter(y_train_res)))

model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)


# Modelo, estrageia combinamos resampling con Smote-Tomek (combinacion de subsampling y oversampling)
os_us = SMOTETomek(ratio=0.5)
X_train_res, y_train_res = os_us.fit_sample(X_train, y_train)

print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))

model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)

# Modelo, estragegia ensamble de modelos con balanceo
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)
#Train the classifier.
bbc.fit(X_train, y_train)
pred_y = bbc.predict(X_test)
mostrar_resultados(y_test, pred_y)


# todos ayudan a mejorar el recall de los fraudes, pero hay que considerar que no haya mucho falsos positivos

