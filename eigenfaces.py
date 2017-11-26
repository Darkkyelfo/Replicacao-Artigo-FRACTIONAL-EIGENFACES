'''
Created on 14 de nov de 2017

@author: raul
'''
from pca import PCA
from copy import deepcopy
import numpy as np
from base import Base
from numba import jit
import math

class Eigenfaces(PCA):
    '''
        Classe que representa a técnica Eigenfaces
    '''
    
    #@jit
    def fit(self,bTreino):
        copia = np.array(deepcopy(bTreino.atributos),dtype='f')
        self.media = np.mean(copia,axis=0)
        n = len(copia)
        copia = copia.T
        self._gerarMatrizSub(copia)
        cov = (1/n)*(copia.T.dot(copia))#matriz de covariancia
        autoValues,autoVectors = np.linalg.eig(cov)
        autoVectors = np.array(autoVectors).T
        autoValues,autoVectors =  zip(*sorted(zip(autoValues, autoVectors),reverse=True))
        self._encontrarAutovetores(copia.T,autoVectors,autoValues, n)
    
    def _gerarMatrizSub(self,matriz):
        colunas = len(matriz[0])
        linhas = len(matriz)
        for j in range(colunas):
            for i in range(linhas):
                #realiza a subtracao do valor - a media ambos elevador a r.
                matriz[i][j] = matriz[i][j] - self.media[i]
                
    def _encontrarAutovetores(self,sub,autoVectores,autoValores,n):
        self.autoVectors = []
        for i,e in enumerate(autoVectores):
            if(autoValores[i]<=0):
                part = 0
            else:
                part = (1/((n*autoValores[i])**(1/2)))
            autoVetor = part*sub.T.dot(e)
            self.autoVectors.append(autoVetor)
        self.autoVectors = np.array(self.autoVectors)
    
    '''
    def _projetar(self, atributorsOri, autoVetores):
        atrR = []
        media = np.mean(atributorsOri,axis=0)
        
        for i in atributorsOri:
            temp = []
            for ind,atr in enumerate(i):
                #temp.append(math.pow(atr, self.r) - math.pow(media[ind],self.r))
                temp.append(atr - self.media[ind])
            atrR.append(temp)
        return super()._projetar(atrR,autoVetores)
    '''
class FractionalEigenfaces(Eigenfaces):
    
    def fit(self,bTreino,r=0.01):
        self.r = r
        super().fit(bTreino)
        
    def _gerarMatrizSub(self,matriz):
        colunas = len(matriz[0])
        linhas = len(matriz)
        for j in range(colunas):
            for i in range(linhas):
                #realiza a subtracao do valor - a media ambos elevador a r.
                matriz[i][j] = math.pow(matriz[i][j],self.r) - math.pow(self.media[i],self.r)
    
    def _projetar(self, atributorsOri, autoVetores):
        atrR = []
        #media = np.mean(atributorsOri,axis=0)
        
        for i in atributorsOri:
            temp = []
            for ind,atr in enumerate(i):
                #temp.append(math.pow(atr, self.r) - math.pow(media[ind],self.r))
                temp.append(math.pow(atr, self.r) - math.pow(self.media[ind],self.r))
            atrR.append(temp)
        return super()._projetar(atrR,autoVetores)
    
