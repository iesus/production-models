'''
Created on Apr 6, 2016

@author: jesus

Contains all methods and classes to perform  the corpus division for cross validation
'''
import cPickle
import numpy as np
import copy,random


'''
 Since we don't have enough sentences, we have no validation set, then, 
 each fold consists of a valtest set and a training set 
'''
class Fold(object):
    
    def __init__(self, train=[],valtest=[]):
        self.trainSet=train
        self.valtestSet=valtest
        
    def saveToPickle(self,filePath):
        outFile=file(filePath,'wb')
        cPickle.dump([self.trainSet,self.valtestSet],outFile,protocol=cPickle.HIGHEST_PROTOCOL)
        outFile.close()
        
    def loadFromPickle(self,filePath):
        inputFile=file(filePath,'rb')
        [self.trainSet,self.valtestSet]=cPickle.load(inputFile)
        inputFile.close()


'''
Receives a fold of situations divided into 10-90 test/train and creates properly the division between conditions
according to the conditions defined in Calvillo et al (2016)
'''
def getFoldTrainingTestSets(rawFoldFileNameActPas,rawFoldFileNameAct,condSize,outFileName):
    
    foldActPas=Fold()
    foldActPas.loadFromPickle(rawFoldFileNameActPas)
    foldAct=Fold()
    foldAct.loadFromPickle(rawFoldFileNameAct)
    
    cond1=foldActPas.valtestSet[:condSize]
    cond2=foldActPas.valtestSet[condSize:condSize*2]
    cond3=foldActPas.valtestSet[condSize*2:]
    
    cond4=foldAct.valtestSet[:condSize]
    cond5=foldAct.valtestSet[condSize:]
    
    trainingSet=[]
    trainTestSet=[]
              
    #cond1 actives are known, passives are asked
    condt1p=[]
    for sit in cond1:
        condt1p.append(sit.passives[0])
        trainingSet.extend(sit.actives)
        trainTestSet.append(sit.actives[0])
    
    #cond2 passives are known, actives are asked
    condt2a=[]
    for sit in cond2:
        condt2a.append(sit.actives[0])
        trainingSet.extend(sit.passives)
        trainTestSet.append(sit.passives[0])
    
    #cond3 dss is completely unknown
    condt3a=[]
    condt3p=[]
    for sit in cond3:  
        condt3a.append(sit.actives[0])
        condt3p.append(sit.passives[0])
    
        
    #cond4 actives are known, passives are asked (but non-existent)
    condt4p=[]
    for sit in cond4:
        exa=sit.actives[0]
        passiveTrainElem=copy.deepcopy(exa)
        passiveTrainElem.active=False
        passiveTrainElem.DSSValue=np.append(sit.value,0.0)
        passiveTrainElem.dss150[150]=0.0#This line is for a corpus with 150dss included
        condt4p.append(passiveTrainElem)
        
        trainingSet.extend(sit.actives)
        trainTestSet.append(sit.actives[0])
    
    #cond5 completely new DSS, actives are asked, passives are asked (but non-existent)
    condt5a=[]
    condt5p=[]
    for sit in cond5:
        condt5a.append(sit.actives[0])
        
        exa=sit.actives[0]
        passiveTrainElem=copy.deepcopy(exa)
        passiveTrainElem.active=False
        passiveTrainElem.DSSValue=np.append(sit.value,0.0)
        passiveTrainElem.dss150[150]=0.0
        condt5p.append(passiveTrainElem)
        
    #ORIGINAL TRAINING SETS
    trainActPas=foldActPas.trainSet
    for sit in trainActPas:
        trainingSet.extend(sit.passives)
        trainTestSet.append(sit.passives[0])
        trainingSet.extend(sit.actives)
        trainTestSet.append(sit.actives[0])
    
    trainAct=foldAct.trainSet
    for sit in trainAct:
        trainingSet.extend(sit.actives) 
        trainTestSet.append(sit.actives[0])     
        
   
    trainList=[trainingSet,trainTestSet]
    testLists=[condt1p,condt2a,condt3a,condt3p,condt4p,condt5a,condt5p]
    
    newFold=Fold(trainList,testLists)
 
    #===========================================================================
    # for lista in testLists:
    #     print
    #     for te in lista:
    #         print te.testItem
    #===========================================================================

    newFold.saveToPickle(outFileName)
    return newFold  
 


'''
Receives a list of elements elemList and creates k files with valtest/train divisions.
Not really used because SetA and SetAP have different sizes
''' 
def getKFolds(k,elemList,seed):

    kFloat=k*1.0
    fullSize=len(elemList)
    valtestSize=int(round(fullSize/kFloat)) #size is proportional to the number of folds
    
    initial=0
    folds=[]
    random.seed(seed)
    random.shuffle(elemList)
    for i in xrange(k):
        valtest=elemList[initial:initial+valtestSize]
        training=elemList[:initial]+elemList[initial+valtestSize:]
        
        fold=Fold(training,valtest)
        fold.saveToPickle("fold_"+str(i)+".pick")
        folds.append(fold)
        
        initial+=valtestSize
    return folds

'''
Receives a list of elements elemList and creates k files with valtest/train divisions.
The size of the valtest set is fixed beforehand
'''
def getKFoldsFixSize(k,valtestSize,elemList,seed,tag):
    
    initial=0
    folds=[]
    random.seed(seed)
    random.shuffle(elemList)
    for i in xrange(k):
        valtest=elemList[initial:initial+valtestSize]
        training=elemList[:initial]+elemList[initial+valtestSize:]
        
        fold=Fold(training,valtest)
        fold.saveToPickle("fold_"+str(i)+"_"+tag+".pick")
        folds.append(fold)
        
        initial+=valtestSize
    return folds

'''
We know that valtestsize is different for the corpus with only actives(28) from the one with passives(42)
So the size is not given as parameter
it is 28 and 42 because each condition contains 14 situations and 
the act set contains 2 conditions, while the actpas contains 3
'''
def getKFoldsFixSizeAPCorpus(k,corpusAP,seed,tag):
    listAct=corpusAP.act
    listActPas=corpusAP.actpas
    getKFoldsFixSize(k,28,listAct,seed,"sitfoldAct_"+tag)
    getKFoldsFixSize(k,42,listActPas,seed,"sitfoldActPas_"+tag)
    
    
'''
Loads a pair of folds (actpasFoldx, actFoldx) and merges them creating the training and testing sets
including the testing conditions
For this case we know that condSize is 14 
'''    
def getKFoldTrainingTestSets(k,tag,condSize,outFileNamePrefix):
    for x in xrange(k):
        
        actpasFileName="fold_"+str(x)+"_sitfoldActPas_"+tag+".pick"
        actFileName="fold_"+str(x)+"_sitfoldAct_"+tag+".pick"
        outputFileName=outFileNamePrefix+"_"+str(x)+".pick"
        
        getFoldTrainingTestSets(actpasFileName,actFileName,condSize,outputFileName)

'''
Takes a CorpusAP object and divides it into K-Folds
'''
def getKFinalTrainTestCondFolds(k,corpusAPFinal,tag,condSize,outTag):
    
    getKFoldsFixSizeAPCorpus(k, corpusAPFinal, 127, tag)#127 is the seed for random
    print "CORPUS DIVIDED INTO K FOLDS"
    
    getKFoldTrainingTestSets(k, tag, condSize,outTag)
    print "EACH FOLD DIVIDED INTO CONDITIONS"
    

if __name__ == '__main__':  
    
    from data.containers import CorpusAP
    corpusAPFinal=CorpusAP()
    corpusAPFinal.loadFromPickle("dataFiles/files-thesis/corpusAPFinal_thesis.pick")
    
    getKFinalTrainTestCondFolds(10,corpusAPFinal,"thesis",14,"trainTest_Cond-thesis")