# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:07:24 2020

@author: Moises New
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#leemos el dataset
df = pd.read_csv("preg3.csv")
n = len(df.index)

#obtencion de 'y' y tranformamos a 0 para + y 1 para -
le = LabelEncoder()
y = df.iloc[:, -1].values
y=le.fit_transform(y)

#como hay datos perdidos y distintos tipos 
#convertimos a tipo numero las 2 columnas A2 y A14 q los reconoce como object
df["A2"] = pd.to_numeric(df["A2"], errors='coerce')
df["A14"] = pd.to_numeric(df["A14"], errors='coerce')

#separamos las columnas numericas y de objetos en dos datas diferentes
#para que la imputacion sea mas facil de realizar
dataNum = df[["A2","A3","A8","A11","A14","A15"]]
dataObj = df[["A1","A4","A5","A6","A7","A9","A10","A12","A13"]]

#imputamos los datos nan en el df numerico con la media
imputacionN = SimpleImputer(missing_values=np.nan,strategy="mean")
dataNumImp = imputacionN.fit_transform(dataNum)

#imputamos los datos '?' en el df de objetos con el mas frecuente
imputacionO = SimpleImputer(missing_values="?",strategy="most_frequent")
dataObjImp = imputacionO.fit_transform(dataObj)

#armamos un nuevo dataset solo para X con los datos ya imputados
X_data={
    "A1": dataObjImp[:,0],  "A2": dataNumImp[:,0],  "A3": dataNumImp[:,1],
    "A4": dataObjImp[:,1],  "A5": dataObjImp[:,2],  "A6": dataObjImp[:,3],
    "A7": dataObjImp[:,4],  "A8": dataNumImp[:,2],  "A9": dataObjImp[:,5],
    "A10": dataObjImp[:,6], "A11": dataNumImp[:,3],"A12": dataObjImp[:,7],
    "A13": dataObjImp[:,8], "A14": dataNumImp[:,4],"A15": dataNumImp[:,5]
    }
data = pd.DataFrame(X_data)

#ahora cambiamos los valores de las letras por valores numericos
#secuenciales de 0 para adelante dependiendo de las cantidad de variable en cada columna
data["A1"] = data["A1"].replace({"a": 0, "b": 1})
data["A4"] = data["A4"].replace({"u": 0, "y": 1, "l": 2, "t": 3})
data["A5"] = data["A5"].replace({"g": 0, "p": 1, "gg": 2})
data["A6"] = data["A6"].replace({"c": 0, "d": 1, "cc": 2, "i": 3, "j": 4,"k": 5, "m": 6, "r": 7, "q": 8, "w": 9, "x": 10, "e": 11, "aa": 12, "ff": 13})
data["A7"] = data["A7"].replace({"v": 0, "h": 1, "bb": 2, "j": 3, "n": 4,"z": 5, "dd": 6, "ff": 7, "o": 8})
data["A9"] = data["A9"].replace({"f": 0, "t": 1})
data["A10"] = data["A10"].replace({"f": 0, "t": 1})
data["A12"] = data["A12"].replace({"f": 0, "t": 1})
data["A13"] = data["A13"].replace({"g": 0, "p": 1, "s": 2})

#ahora lo pasamos a X como una matriz
X = np.array(data[["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11",
                   "A12","A13","A14","A15"]])

#teniendo X y Y entonces separamos en 80% para aprendizaje y 20% para testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#procedemos a entranar a la red neuronal con X_train, y_train
clasificador = MLPClassifier(tol=1e-2)
clasificador.fit(X_train, y_train)

#predecimos al 20% 
y_pred = clasificador.predict(X_test)
print("Prediccion del 20% del dataset")
print(y_pred)

#armamos la matriz de confucion 
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusion")
print(cm)
