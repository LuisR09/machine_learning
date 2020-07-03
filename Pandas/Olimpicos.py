# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:03:12 2020

@author: USUARIO
"""

import pandas as pd
df = pd.read_csv("JuegosOlimpicosVerano.csv")

df.shape

df.info()

df.describe(include="all")

df.columns

# Información individual de una columna
df["Medal"].describe()

df["Gender"].describe()

df["Country"].describe()

df["Gender"].value_counts()

grupo_genero = df.groupby("Gender")
grupo_genero.describe()

grupo_medal = df.groupby("Medal")
grupo_medal.describe()


cantidad_men = df.groupby("Gender")["Year"].count()["Men"]
print(cantidad_men)


new_dataset = df[["Gender", "Year", "Country"]]
df2 = pd.DataFrame(df, columns=["Gender", "Year", "Country"])


# Valores unicos
pd.unique(df["Country"])


# Filtrar filas de los años 2000 a 2012 y de USA (dos formas)
df2[(df2.Year>=2010) & (df2.Year<=2012) & (df2.Country=="USA")]
df2[(df2["Year"]>=2010) & (df2["Year"]<=2012) & (df2["Country"]=="USA")]




