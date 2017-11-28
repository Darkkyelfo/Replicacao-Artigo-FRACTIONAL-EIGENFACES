'''
Created on 24 de nov de 2017

@author: raul
'''
from base import Base
import cv2
import glob as g

#Essa funcao transforma todo um diretorio de imagens em uma base
def imgsParaBase(caminho,dirClasse = "class_",qtClasses=15,tipoArq = "png"):
    classes = []
    atributos = []
    for i in range(1,qtClasses+1):
        t = str(i)
        if(len(t)==1):
            t = "0" + t
        find = "%s/%s%s/*.%s"%(caminho,dirClasse,t,tipoArq)
        imagens = g.glob(find)
        for caminhoImg in imagens:
            atr = img2Dto1D(caminhoImg)
            classes.append(i-1)
            atributos.append(atr)
    return Base(classes,atributos)
            


def img2Dto1D(caminhoImg):
    img = cv2.imread(caminhoImg,0)
    array1D = img.reshape(img.shape[0]*img.shape[1])
    return array1D

    