# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:23:28 2020

@author: USUARIO
"""

# Importancia de escalar las vbles

# Importing pandas
import pandas as pd
from sklearn.metrics import accuracy_score

# importing training data set
X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")

# Importing testing data set
X_test = pd.read_csv("X_test.csv")
Y_test = pd.read_csv("Y_test.csv")

X_train.head()

# Histograms
X_train.hist(figsize=[11, 11])  # hay datos que están en una escala muy mayor a otros

# Sin escalar los datos
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                 "Credit_History"]], Y_train.values.ravel())

Y_predict = knn.predict(X_test[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                 "Credit_History"]])

knn.score(X_test[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                 "Credit_History"]], Y_test)

accuracy_score(Y_test, Y_predict) # igual que el anterior #=61%


Y_train.Target.value_counts()/Y_train.Target.count()

# Con escalar los datos
from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()

X_train_minmax = min_max.fit_transform(X_train[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                 "Credit_History"]])

X_test_minmax = min_max.fit_transform(X_test[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                 "Credit_History"]])

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax, Y_train)

accuracy_score(Y_test, knn.predict(X_test_minmax)) #=75%

# Al escalar los datos aumenta la precisión del modelo de 61 a 75%



# Estandarizarl los datos
from sklearn.preprocessing import scale
X_train_scale = scale(X_train[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                 "Credit_History"]])
X_test_scale = scale(X_test[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                 "Credit_History"]])

# Aplicando regresión logística no mejora mucho
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(penalty = "l2", C=0.01)  # l2 asume que la media es cero y la desviación por el mismo orden
log.fit(X_train_scale, Y_train)

accuracy_score(Y_test, log.predict(X_test_scale))
# Al estandarizar los datos mejora el modelo, ya que usando "l2" los datos ya están en su media cero




# Codificar los datos categóricos
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in X_test.columns:
    if X_test[i].dtypes=="object":
        data = X_train[i].append(X_test[i])
        le.fit(data.values)
        X_train[i]=le.transform(X_train[i])
        X_test[i]=le.transform(X_test[i])
     
     
from sklearn.tree import DecisionTreeClassifier
algoritmo = DecisionTreeClassifier().fit(X_train, Y_train)

accuracy_score(Y_test, algoritmo.predict(X_test))
# agregando las columnas de datos categóricos no se ve ninguna mejora, significa que esos datos categóricos no son significativos



