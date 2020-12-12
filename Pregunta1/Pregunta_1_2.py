# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:31:44 2020

@author: Moises New
"""
import pandas as pd
import numpy as np

def vecino_mas_cercano(estado_inicial, matriz_costos):
    ''' vecino mas cercano, recibe un estado inicial, una matriz de costos y devuelve un tour '''
    # Declara e inicializa candidatos, estado y tour
    candidatos = list(range(len(matriz_costos)))
    candidatos.remove(estado_inicial)
    estado = estado_inicial
    tour = [estado]
    # Ciclo de busqueda termina cuando no hay más candidatos
    while True:
        if len(candidatos) == 0:
            tour.append(estado_inicial)
            break
        # Selecciona el candidato con el menor costo
        estado = min(candidatos, key=(lambda j: matriz_costos[estado][j] if estado >= j else matriz_costos[j][estado]))
        # Agrega el estado al tour y lo remueve de la lista de candidatos
        tour.append(estado), candidatos.remove(estado)
    return tour

def costo_tour(tour, matriz_costos):
    ''' costo del tour, recibe el tour, la matriz de costos y devuelve el costo del tour(menor fitness)'''
    costo = sum(matriz_costos[tour[i]][tour[i+1]] 
        if tour[i] >= tour[i+1]
        else matriz_costos[tour[i+1]][tour[i]]
        for i in tour[:-1])
    return costo

def main():
    df = pd.read_csv("costos.csv")
    print(df)
    #Convirtiendo el df a matriz
    matriz = np.array(df)
    av = {
      "Lugares" : len(matriz),#Se obtiene la cantidad de lugares que se visitará
      "Matriz_costos" : matriz  
    }
    costos = av["Matriz_costos"]
    #le mandamos el estado de donde comenzara el tour
    tour = vecino_mas_cercano(0, costos)
    print("Tour:", tour)
    print("Costo: ", costo_tour(tour, costos)) 

if __name__ == '__main__':
    main()
    
