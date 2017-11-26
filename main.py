from funcoesexecucao import execucaoEigen,execucaoFracEigen,execucaoFracPCA,executarHoldSo
from multiprocessing import Pool

hold = 10
r=0.01
pEigen = [("orl", "EigenOrl.txt",40, hold),("yale","EigenYale.txt",15,hold),("umist", "EigenUmist.txt", 20, hold)]
pFracEigen = [("yale","EigenFracYale.txt",15,r,hold),("orl", "EigenFracOrl.txt",40,r, hold),("umist", "EigenFracUmist.txt",r,20, hold)]
#pFracPca = [("yale","FracPCAYale.txt",15,r,hold),("orl", "FracPCAOrl.txt",40,r, hold),("umist", "FracPCAUmist.txt", 20,r, hold)]
def executarPoolFracEigen(args):
    execucaoFracEigen(*args)
'''
def executarPoolFracPCA(args):
    execucaoFracPCA(*args)
'''
def executarPoolEigen(args):
    execucaoEigen(*args)

#executa os experimentos paralelamente
#poolEigen = Pool(3)
#poolEigen.map(executarPoolEigen,pEigen)
#poolFracEigen = Pool(3)
#poolFracEigen.map(executarPoolFracEigen,pFracEigen)
#execucaoEigen("yale_risized","EigenYaleRisized.txt",15,hold)
#execucaoFracEigen("yale_risized","EigenFracYaleRisized.txt",15,r,hold)
execucaoFracPCA("yale_risized","pcaFracYaleRisized.txt",15,r,hold)
#execucaoEigen("yale","EigenYale.txt",15,hold)
#execucaoFracEigen("yale","EigenFracYale.txt",15,r,hold)
#cria um pool de processos para o PCA fracionario
#poolFracPCA = Pool(3)
#poolFracPCA.map(executarPoolFracPCA,pFracPca)




