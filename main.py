from funcoesexecucao import execucaoEigen,execucaoFracEigen,execucaoFracPCA
from multiprocessing import Pool

hold = 10
r = 0.01
pEigen = [("yale","EigenYale.txt",15,hold),("orl", "EigenOrl.txt",40, hold),("umist", "EigenUmist.txt", 20, hold)]
pEigenRisized = [("yale_risized","EigenYaleRisized.txt",15,hold),("orl_risized", "EigenOrlRisized.txt",40, hold),("umist_risized", "EigenUmistRisized.txt", 20, hold)]
pFracEigenRisized = [("yale_risized","fracEigenYaleRisized.txt",15,r,hold),("orl_risized", "fracEigenOrlRisized.txt",40,r,hold),("umist_risized", "fracEigenUmistRisized.txt", 20,r,hold)]
pFracEigen = [("yale","fracEigenYale.txt",15,r,hold),("orl", "fracEigenOrl.txt",40,r, hold),("umist", "fracEigenUmist.txt",r,20, hold)]
pFracPca = [("yale_risized","FracPCAYaleRisized.txt",15,r,hold),("orl_risized", "FracPCAOrlRisized.txt",40,r,hold),("umist_risized", "FracPCAUmistRisized.txt", 20,r,hold)]


def executarPoolFracEigen(args):
    execucaoFracEigen(*args)

def executarPoolFracPCA(args):
    execucaoFracPCA(*args)

def executarPoolEigen(args):
    execucaoEigen(*args)

#executa os experimentos paralelamente

'''
poolEigenR = Pool(3)
poolEigenR.map(executarPoolEigen,pEigenRisized)

poolEigenFracR = Pool(2)
poolEigenFracR.map(executarPoolFracEigen,pFracEigenRisized)

poolFracPCA = Pool(2)
poolFracPCA.map(executarPoolFracPCA,pFracPca)

    
execucaoFracEigen("yale_risized","fracEigenYaleRisized.txt",15,r,hold)

poolEigen = Pool(3)
poolEigen.map(executarPoolEigen,pEigen)
'''
'''
poolEigenFrac = Pool(3)
poolEigenFrac.map(executarPoolFracEigen,pFracEigen)
'''
execucaoEigen("umist", "EigenUmist.txt", 20, hold)
execucaoFracEigen("umist", "FracEigenUmist.txt",20,r,hold)
execucaoFracPCA("yale_risized","FracPCAYaleRisized.txt",15,r,hold)
execucaoFracPCA("orl_risized", "FracPCAOrlRisized.txt",40,r,hold)
execucaoFracPCA("umist_risized", "FracPCAUmistRisized.txt",20,r,hold)

#poolFracEigen = Pool(3)

#poolFracPCA = Pool(3)
#poolFracPCA.map(executarPoolFracPCA,pFracPca)




