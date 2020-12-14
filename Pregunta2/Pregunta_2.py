# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:51:47 2020

@author: Moises New
"""

import random

modelEnd = [1,1,1,1,1,1,1,1,1,1] 
tamIndividuo = 10 

num = 100 #Cantidad de individuos
generation = 3 #Generaciones
presicion = 3 #individuo>2
mutacion_chance = 0.2

def individuo(min, max):
    return[random.randint(min, max) for i in range(tamIndividuo)]

def newpoblacion():
    return [individuo(0,1) for i in range(num)]

# Funcion la que se debe cambiar en funcion a f(x)
def functionType(individuo):
    dec = int("".join(map(str,individuo)),2)
    dec = dec**3 + dec**2 + dec **1
    return dec

def selection_reproduction(poblacion):
    evaluacion = [ (functionType(i), i) for i in poblacion]
    evaluacion = [i[1] for i in sorted(evaluacion)]
    poblacion = evaluacion
    selected = evaluacion[(len(evaluacion)-presicion):]
    print("---",selected)
    for i in range(len(poblacion)-presicion):      
        puntoCambio = random.randint(1,tamIndividuo-1)
        padre = random.sample(selected, 2)
        poblacion[i][:puntoCambio] = padre[0][:puntoCambio]
        poblacion[i][puntoCambio:] = padre[1][puntoCambio:]      
    return poblacion

def mutacion(poblacion):
    for i in range(len(poblacion)-presicion):
        if random.random() <= mutacion_chance: 
            puntoCambio = random.randint(1,tamIndividuo-1) 
            new_val = random.randint(0,9) 
            while new_val == poblacion[i][puntoCambio]:
                new_val = random.randint(0,9)
            poblacion[i][puntoCambio] = new_val
    return poblacion

def main():
    poblacion = newpoblacion()
    #print(poblacion)
    print("\npoblacion inicial:\n%s"%(poblacion))
    poblacion = selection_reproduction(poblacion)
    print("\Selection poblacion:\n%s"%(poblacion))
    poblacion = mutacion(poblacion)
    print("\mutacion poblacion:\n%s"%(poblacion))
    
if __name__ == '__main__':
    main()

