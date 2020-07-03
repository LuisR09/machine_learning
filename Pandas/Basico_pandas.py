# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:51:10 2020

@author: USUARIO
"""

#### Uso básico de pandas ####

import pandas as pd

# creamos un dataset

df = pd.DataFrame(data={"Pais":["Mexico","Argentina","Espana","Colombia"],
                        "Poblacion":[127212000,45167000,47099000,48922000]})


# Ordenamos por columna
df.sort_values(["Poblacion"], ascending=False)

df = df.sort_values(["Pais"])

# Agregar una columna
df["Superficie"] = [1964375,2780400,505944,1142748]

# Asignar un valor a toda una columna nueva
df["Deporte"] = "Futbol"

# Eliminar una columna
df = df.drop(["Deporte"],axis=1)

# Eliminar multiple columnas
df.drop(["Superficie","Pais"], axis=1)

# Contar las filas
cantidad_filas = len(df)

# Agregar una fila nueva al final
df.loc[4] = ["Benezuela", 0, 916445]

# Actualiar fila entera
df.loc[4] = ["Venezuela", 0, 916445]

# Actualizo una celda
df.at[4,"Poblacion"] = 32423000

# Eliminar una fila
df.drop([3])

# Eliminar multiples filas
df.drop([3,1])


#### Filtrar ####
# Países con mas de 46 millones de habitantes
mas_de_46 = df[df["Poblacion"] > 46000000]

# Mas de 46 mill y superficie menor a 600.000 km2
doble_filtro = df[(df["Poblacion"] > 46000000) & (df["Superficie"] < 600000)]

# Busco por un valor específico
por_nombre = df[df["Pais"] == "Colombia"]

# Paises con nombre mayor a 6 letras
nombre_largo = df[df["Pais"].str.len() > 6]

# Filtrar por True/False
arreglo = [True, False, False, False, True] 
df[arreglo]

# Obtener el indice de una fila
por_nombre = df[df["Pais"] == "Colombia"]
por_nombre.index.tolist()[0]

#### Aplicar operaciones entre columnas ####
# agregamos en una nueva columna el ratio de habitantes por superficie
df["Habit_x_km2"] = df["Poblacion"]/df["Superficie"]
df.sort_values(["Habit_x_km2"])

# Aplicar una operación definida
def crear_codigo(name):
    name = name.upper()
    name = name[0:4]
    return name

# aplicamos usando 1 columna
df["Codigo"] = df["Pais"].apply(crear_codigo)


# Aplicamos una funcion a cada fila
def categoria(fila):
    pob = fila["Poblacion"]
    habit = fila["Habit_x_km2"]
    if pob > 46000000:
        if habit < 50:
            return "A"
        else:
            return "B"
    return "C"
    
df["Categoria"] = df.apply(categoria, axis=1)


# Aplicar enviando algunas columnas como parámetros
def asigna_color(codigo, categoria):
    if categoria=="A":
        return "rojo"
    if codigo=="ESPA":
        return "verde"
    return "azul"

df["color"] = df.apply(lambda x: asigna_color(x["Codigo"], x["Categoria"]), axis=1)

# Mapeo
df["mapeo_color"] = df["color"].map({"azul":0, "rojo":1, "verde":2})

# Reordenamos columnas
df = df[["Codigo","Pais","Poblacion","Categoria","Superficie","Habit_x_km2"]]


#### Join entre tablas ####
# concatenar usando indice

# creamos un DF nuevo, le asignamos el Codigo como identificador único
df_comida = pd.DataFrame(data={
                        "Comida":["Burritos","Milanesa","Tortilla","Sancocho","Arepas"]},
                        index = ["MEXI","ARGE","ESPA","COLO","VENE"])

# Asignamos indice en nuestro DF inicial
df_index = df.set_index("Codigo")

# Hacemos el join por indice
result1 = pd.concat([df_index, df_comida], axis=1, sort=True)

# Left join por columna clase(merge)
# imaginemos que tenemos un DF nuevo, le asignamos el codigo como identificador único
df_factor = pd.DataFrame(data={"Categoria":["A","B","C"],
                               "Factor":[12.5,103,0.001]})

result2 = pd.merge(df, df_factor, how="left", on=["Categoria"])


# Adicionar multiples filas desde otra tabla con Append
# Supongamos que tenemos otra tabla:
df_otros = pd.DataFrame(data={"Pais":["Brasil","Chile"],
                              "Poblacion":[210688000, 19241000],
                              "Superficie":[8515770,56102]})

# Queremos agregar estas filas al final
df.append(df_otros, ignore_index=True, sort=True)

# Agrupar
# agrupo por categoria y sumo cuantos hay de cada una
grupo2 =df.groupby(["Categoria"]).size()

# Agrupo por categoria y sumo
grupo1 = df.groupby(["Categoria"]).sum()

# Agrupamos por 2 variables y sumarizamos
tabla = result2[['Categoria', 'Factor']].groupby(['Categoria'], as_index=False).agg(['mean', 'count', 'sum'])

# Pivotar una tabla
tabla_t = pd.pivot_table(result2, index="Categoria", columns="Pais", values="Factor").fillna(0)

# Transponer una tabla
df.T

#### Visualizacion ####

df[["Poblacion", "Superficie"]].plot.hist(bins=5, alpha=0.5)

df.set_index("Pais")["Poblacion"].plot(kind="bar")

df.set_index("Pais")["Habit_x_km2"].plot(kind="area")

df.set_index("Pais").plot.barh(stacked=True)

df.set_index("Pais")["Superficie"].plot.pie()

df.plot.scatter(x="Habit_x_km2", y="Superficie")

df.info()

df.describe()

df.shape

# Iterar un dataframe
for index, row in df.iterrows():
    print(row["Pais"])

# Obtener sumatoria, media y cantidad de una columna
print(df["Habit_x_km2"].mean())
print(df["Habit_x_km2"].sum())
print(df["Habit_x_km2"].count())
print(df["Habit_x_km2"].min())
print(df["Habit_x_km2"].max())

# Revisar si tenemos nulos en la tabla
df.isnull().sum()

# Acceso a una columna
df.Superficie

# unicos
df.Categoria.unique()
# cantidad de unicos
len(df.Categoria.unique())

# Contabilizar por una columna
pd.value_counts(df["Categoria"], sort = True)

# Obtener ultima fila
df.iloc[-1]

# Obtener primera columna
df.iloc[:, 2]

# Obtener una columna dentro de una condicion
df.loc[df["Superficie"] < 1000000, ["Pais"]]

# Modificar un valor con loc
df.loc[df["Superficie"] < 1000000,"Categoria"] = "D"
