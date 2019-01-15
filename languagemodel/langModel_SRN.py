import theano, numpy, random, os, sys
import matplotlib.pyplot as plt

import rnn.langmodelSRNN_noBPTT
from data.containers import CorpusLM
from tools.plusplus import xplusplus

sys.path.append("../data")
outputsPath="../outputs"
wordLocalistMapPath='../data/dataFiles/map_localist_words.txt'

#corpusFilePath="../data/dataFiles/filesSchemas_with150DSS_withSims96/corpus_UID_imbalancedProbs.pick"
corpusFilePath="../data/dataFiles/files-thesis/corpusUID-thesis.pick"


def localistToIndices(localistMatrix):
    return [numpy.argmax(localist) for localist in localistMatrix]

def indicesToLocalist(indices,vocabSize):
    localistMatrix=[]
    for wordIndex in indices:
        localistVector=numpy.zeros(vocabSize, dtype=theano.config.floatX)  # @UndefinedVariable
        localistVector[wordIndex]=1
        localistMatrix.append(localistVector)
    return localistMatrix

def getFolders(outputsPath, params):
    """
    Creates the 3 folders where all results/models will be stored
    folderThisRun: folder containing all the files of this particular run, will be contained in folderThisModel which contains all runs of this specific python file
    bestModel: parameters that achieved best performance on the training set
    lastModel: parameters that the model has at the end of training
    """
    #Create folder that contains all the runs for this python file
    currentFolder=outputsPath+"/"+os.path.basename(__file__).split('.')[0]
    folderThisModel=currentFolder+"_outputs"

    if not os.path.exists(folderThisModel): os.mkdir(folderThisModel)
    
    #Create folder for all the files of this specific run
    folderThisRun=folderThisModel+"/output"
    folderThisRun+="_"+str(params['nhidden'])+"h"
    folderThisRun+="_"+str(params['vocab_size'])+"voc"
    folderThisRun+="_"+str(params['lr'])+"lr"
    folderThisRun+="_"+str(params['nepochs'])+"ep"
    folderThisRun+="_"+params['label']
    
    if not os.path.exists(folderThisRun): os.mkdir(folderThisRun)
    
    #Create folder for plots
    plotsFolder=folderThisRun+"/plots"
    if not os.path.exists(plotsFolder): os.mkdir(plotsFolder)
    
    #Create folders for best and last model parameters
    bestModel=folderThisRun+"/bestModel"
    if not os.path.exists(bestModel): os.mkdir(bestModel)
    lastModel=folderThisRun+"/lastModel"
    if not os.path.exists(lastModel): os.mkdir(lastModel)
    
    return folderThisRun,bestModel,lastModel,plotsFolder


if __name__ == '__main__':

    if len(sys.argv)>1:
        x=1
        s={
             'lr':float(sys.argv[xplusplus("x")]), #learning rate
             'decay':int(sys.argv[xplusplus("x")]), # whether decay on the learning rate if improvement stops
             'nhidden':int(sys.argv[xplusplus("x")]), # number of hidden units
             'vocab_size':int(sys.argv[xplusplus("x")]),
             'seed':int(sys.argv[xplusplus("x")]),    
             'nepochs':int(sys.argv[xplusplus("x")]),
             
             'label':sys.argv[xplusplus("x")],#"dotsPaper_dssSeparated_811_1",
             'load':int(sys.argv[xplusplus("x")]), #Whether the model is already trained or not, 
             'inputFile':sys.argv[xplusplus("x")] #FILE containing the corpus input
             
         }       
    else:
        s = {
         'lr':0.24, #learning rate 0,124
         'decay':True, # decay on the learning rate if improvement stops
         'nhidden':50, # number of hidden units
         'vocab_size':43,
         'seed':345,
         'nepochs':200,
         
         'label':"uid-thesiscorpus",#unnormalized2-1-1540
         'load':True, #Whether the model is already trained or not
         'inputFile':corpusFilePath #FILE containing the corpus input
         }
    
    #LOAD FILES
    corpusLM=CorpusLM()
    corpusLM.loadFromPickle(s['inputFile'])
    
    trainList=corpusLM.training
    validateList=corpusLM.validation
    testLists=corpusLM.testing
    
    
    #CREATE SRNN AND INITIALIZE VARS
    srnn = rnn.langmodelSRNN_noBPTT.model(
                              inputDimens=s['vocab_size'],
                              hiddenDimens = s['nhidden'],
                              outputDimens= s['vocab_size'],
                              startSentIndex=-1 #index of the sentence boundary that is used as input at t=0
                     )       
    folderThisRun,bestModel,lastModel,plotsFolder=getFolders(outputsPath,s)
    random.seed(s['seed']) 

    #IF THE MODEL HASNT BEEN TRAINED 
    if not s['load']:   
        
        trainListLocalist=[item.wordsLocalist for item in trainList]
        validateListLocalist=[item.wordsLocalist for item in validateList]
        
        outputFile= open(folderThisRun+'/output.txt', 'w+')
        best_valError = numpy.inf
        bestEp=0
        epErrors=[]
        epValErrors=[]         
        current_LR = s['lr']
        
        for epoch in xrange(s['nepochs']):
            random.shuffle(trainListLocalist)

            errors=srnn.epochTrain(trainListLocalist,current_LR)
            epErrors.append(sum(errors))
            
            valErrors=srnn.epochValidate(trainListLocalist) 
            #WE VALIDATE USING TRAINLIST BECAUSE THE VALIDATION SET IS NOT REALLY SET FOR LANGUAGE MODELLING BUT FOR TESTING UID
            valError=sum(valErrors)
            epValErrors.append(valError)
            
            outputLine='Epoch: '+str(epoch)+' lr: '+str(current_LR)+' valError: '+str(valError)
            
            if valError < best_valError:
                srnn.save(bestModel)
                best_valError = valError
                bestEp=epoch
                lastChange_LR=epoch#last time the learning rate changed,reset to current epoch after an improvement
                
                outputLine= 'NEW BEST '+outputLine                       
            
            print outputLine
            outputFile.write(outputLine+'\n')
                    
            errorsPlot=plt.figure(100000)
            plt.plot(epErrors)
            plt.savefig(folderThisRun+"/errorsTrainEp.png")
                          
            valErrorPlot=plt.figure(1000000)
            plt.plot(epValErrors)
            plt.savefig(folderThisRun+"/valErrors.png")
                
            # learning rate halves if no improvement in 10 epochs
            if s['decay'] and (epoch-lastChange_LR) >= 10: 
                current_LR *= 0.5
                lastChange_LR=epoch
                
            #TRAINING STOPS IF THE LEARNING RATE IS BELOW THRESHOLD OR IF NO IMPROVEMENT DURING 30 EPOCHS
            if current_LR < 1e-3 or (epoch-bestEp)>=30:     
                break  
            
        srnn.save(lastModel)
        
        outputLine='BEST RESULT: epoch '+str(bestEp)+' valError: '+str(best_valError)+' with the model '+folderThisRun
        print outputLine
        outputFile.write(outputLine)
        outputFile.close()
      
    else: #IF THE MODEL WAS ALREADY TRAINED AND WE ARE ONLY LOADING IT FOR TESTING
        srnn.load(bestModel)

        for sent in validateList[:5]:
            
            print sent.testItem
            sentenceWords=sent.testItem.split()
            
            probLangModel=srnn.getSentenceWordProbs(sent.wordsLocalist, sentenceWords)
            print probLangModel
            
            print 
            print


       
