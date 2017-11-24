'''
Created on 14 de nov de 2017

@author: raul
'''
from pca import PCA
from copy import deepcopy
import numpy as np
from base import Base
from numba import jit

class Eigenfaces(PCA):
    '''
        Classe que representa a técnica Eigenfaces
    '''
    
    #@jit
    def fit(self,bTreino):
        copia = np.array(deepcopy(bTreino.atributos),dtype='f')
        media = np.mean(copia,axis=0)
        n = len(copia)
        copia = copia.T
        colunas = len(copia[0])
        linhas = len(copia)
        for j in range(colunas):
            for i in range(linhas):
                copia[i][j] = copia[i][j] - media[i]
        cov = (1/n)*(copia.T.dot(copia))#matriz de covariancia
        autoValues,autoVectors = np.linalg.eig(cov)
        autoVectors = np.array(autoVectors).T
        autoValues,autoVectors =  zip(*sorted(zip(autoValues, autoVectors),reverse=True))
        self.__encontrarAutovetores(copia.T,autoVectors,autoValues, n)
        
    def __encontrarAutovetores(self,sub,autoVectores,autoValores,n):
        self.autoVectors = []
        for i,e in enumerate(autoVectores):
            autoVetor = (1/((n*autoValores[i])**(1/2)))*sub.T.dot(e)
            self.autoVectors.append(autoVetor)
        self.autoVectors = np.array(self.autoVectors)
            
    
class FractionalEigenfaces(Eigenfaces):
    
    #@jit
    def fit(self,bTreino,r=0.01):
        copia = np.array(deepcopy(bTreino.atributos),dtype='f')
        media = np.mean(copia,axis=0)
        n = len(copia)
        copia = copia.T
        colunas = len(copia[0])
        linhas = len(copia)
        for j in range(colunas):
            for i in range(linhas):
                copia[i][j] = copia[i][j]**r - media[i]**r
        cov = (1/n)*(copia.T.dot(copia))#matriz de covariancia
        autoValues,autoVectors = np.linalg.eig(cov)
        autoVectors = np.array(autoVectors).T
        autoValues,autoVectors =  zip(*sorted(zip(autoValues, autoVectors),reverse=True))
        self.__encontrarAutovetores(copia.T,autoVectors,autoValues, n)
        

