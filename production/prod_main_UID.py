'''
Created on May 6, 2016
'''
import numpy, random
import os,sys

import matplotlib.pyplot as plt
import data.loadFiles as loadFiles
from data.containers import CorpusLM
from tools.similarities import levenSimilarity
from tools.plusplus import xplusplus
from production.decoder import SentenceDecoderUID
from languagemodel.langModel_SRN import indicesToLocalist

import rnn.prodSRNN_UID 
import rnn.langmodelSRNN_noBPTT

sys.path.append("../data")
corpusFilePath="../data/dataFiles/filesSchemas_with150DSS_withSims96/corpusUID.pick"

wordLocalistMapPath='../data/dataFiles/map_localist_words.txt'
dsssMatrixPath="../data/dataFiles/model_vectors"
outputsPath="../outputs"

modelProductionPath="../outputs/prod_main_mon_5cond_outputs/output_beliefVector_120h_0.24lr_200ep_dots_15_40_monitor_sigm_0/lastModel"
modelSurprisalPath="../outputs/langModel_UIDPaper_outputs/output_beliefVector_100h_0.24lr_200ep_dots_15_40_monitor_sigm_0/lastModel"


def mostSimilarEquivalentsLevens(trainingElement,modelProduction):
    equivalentsIndices=[localistToIndices(equivalent.wordsLocalist) for equivalent in trainingElement.equivalents]
    similarities=[levenSimilarity(eq,modelProduction) for eq in equivalentsIndices]
    mostSimilar=numpy.argmax(similarities, 0)
    
    return (similarities[mostSimilar],equivalentsIndices[mostSimilar])

def localistToIndices(localistMatrix):
    return [numpy.argmax(localist) for localist in localistMatrix]

def indicesToWords(indices,indexWordMapping):
    return [indexWordMapping[index] for index in indices]

    
def evaluateSRNNUID(srnnRR,outFile, evalSet,periods):
    predictions_test=srnnRR.getModelPredictions(evalSet,periods)
    
    simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(evalSet,predictions_test)]
    similarities=[acc for (acc,gold) in simgolds]
    golds=[gold for (acc,gold) in simgolds]
    
       
    predWords=[indicesToWords(pred,mapIndexWord) for pred in predictions_test]
    labelWords=[indicesToWords(label,mapIndexWord) for label in golds]
    
    printResults(outFile,predWords,labelWords,similarities,evalSet)

def printResults(outFile, predWords,labelWords, similarities, evalSet):
    accuracyGlobal=numpy.sum(similarities)/len(similarities)
    
    perfect=[]
    almostPerfect=[]
    mildlyBad=[]
    worst=[]
    
    def printSubSet(label,setValues,superSize):
        print label
        outFile.write(label+"\n")
        
        for (pw,lw,_) in setValues:
            print pw,lw
            outFile.write(str(pw)+" "+str(lw)+"\n")
            
        print len(setValues)  #number of sentences that fell under this range
        outFile.write(str(len(setValues))+"\n")
        print len(setValues)/float(superSize)#proportion of these sentences with respect to the whole condition
        outFile.write(str(len(setValues)/float(superSize))+"\n\n")
        print 
    
    for pw, lw, acc,item in zip(predWords, labelWords, similarities, evalSet):
            print item.testItem
            print pw,lw
            print acc 
            print
            outFile.write(item.testItem+"\n")
            outFile.write(str(pw)+" "+str(lw)+"\n")
            outFile.write(str(acc)+"\n\n")
            
            if acc==1.0: perfect.append((pw,lw,acc))
            elif acc>=0.8: almostPerfect.append((pw,lw,acc))
            elif acc>=0.5: mildlyBad.append((pw,lw,acc))
            else: worst.append((pw,lw,acc))
            
    printSubSet("PERFECT INSTANCES",perfect,len(evalSet))
    printSubSet("ALMOST PERFECT",almostPerfect,len(evalSet))
    printSubSet("MILDLY BAD", mildlyBad,len(evalSet))
    printSubSet("WORST INSTANCES", worst,len(evalSet))
              
    print   
    print accuracyGlobal
    outFile.write("\n"+str(accuracyGlobal)+"\n")
    
def getFolders(outputsPath, params):
    #Create folder that contains all the runs for this python file
    currentFolder=outputsPath+"/"+os.path.basename(__file__).split('.')[0]
    folderThisModel=currentFolder+"_outputs"

    if not os.path.exists(folderThisModel): os.mkdir(folderThisModel)
    
    #Create folder for all the files of this specific run
    folderThisRun=folderThisModel+"/output"
    
    folderThisRun+="_"+params['inputType']
    folderThisRun+="_"+str(params['nhiddenLM'])+"hLM"
    folderThisRun+="_"+str(params['nhiddenRR'])+"hRR"
    folderThisRun+="_"+str(params['lr'])+"lr"
    folderThisRun+="_"+str(params['nepochs'])+"ep"
    if params['periods']: folderThisRun+="_dots"
    folderThisRun+="_"+params['label']
    
    if not os.path.exists(folderThisRun): os.mkdir(folderThisRun)
    
    #Create folders for best and last model parameters
    bestModel=folderThisRun+"/bestModelUID"
    if not os.path.exists(bestModel): os.mkdir(bestModel)
    lastModel=folderThisRun+"/lastModelUID"
    if not os.path.exists(lastModel): os.mkdir(lastModel)
    
    return folderThisRun,bestModel,lastModel


if __name__ == '__main__':

    if len(sys.argv)>1:
        x=1
        s={
             'lr':float(sys.argv[xplusplus("x")]), #learning rate
             'verbose':int(sys.argv[xplusplus("x")]),
             'decay':int(sys.argv[xplusplus("x")]), # decay on the learning rate if improvement stops
             'nhiddenLM':int(sys.argv[xplusplus("x")]), # number of hidden units for language layer
             'nhiddenRR':int(sys.argv[xplusplus("x")]), # number of hiiden units for reranker layer
             'seed':int(sys.argv[xplusplus("x")]),
             
             'percTrain':float(sys.argv[xplusplus("x")]),
             'percValidate':float(sys.argv[xplusplus("x")]),
             'percTest':float(sys.argv[xplusplus("x")]),
             
             'nepochs':int(sys.argv[xplusplus("x")]),
             'label':sys.argv[xplusplus("x")],#"dotsPaper_dssSeparated_811_1",
             'periods':int(sys.argv[xplusplus("x")]),
             'loadRR':int(sys.argv[xplusplus("x")]),#whether the reranker model was already trained or not
             'inputType':sys.argv[xplusplus("x")], #dss or sitVector or compVector
             'actpas':sys.argv[xplusplus("x")],
             'inputFile':sys.argv[xplusplus("x")], #FILE containing the corpus input
             'lmFolder':sys.argv[xplusplus("x")] #folder containing the already trained production network      
         }
           
    else:
        s = {
         'lr':0.24, #learning rate
         'verbose':1,
         'decay':True, # decay on the learning rate if improvement stops
         'nhiddenLM':120, # number of hidden units
         'nhiddenRR':30, # number of hidden units
         
         'seed':345,
         
         'percTrain':0.9,
         'percValidate':0.0,
         'percTest':0.1,
         
         'nepochs':80,
         'label':"15_40_trainstop_monitor_25K_UID_sigm_paper_ttesteset1",#unnormalized2-1-1540
         'periods':True,
         'loadRR':True,
         'inputType':'beliefVector', #dss or sitVector or compVector
         'actpas':True,
         'inputFile':corpusFilePath, #FILE containing the corpus input
         'lmProductionFolder':modelProductionPath#"/Users/jesus/Documents/EclipseWorkspace/productionModel/outputs/prod_main_mon_5cond_outputs/output_beliefVector_120h_0.124lr_200ep_dots_15_40_trainstop_monitor_25K_sch/lastModel"
         }

    if s['periods']: s['vocab_size']=43
    else: s['vocab_size']=42
    
    if s['inputType']=='sitVector' or s['inputType']=='compVector' or s['inputType']=="beliefVector": s['inputDimension']=44
    if s['inputType']=='dss': s['inputDimension']=150
    if s['actpas']:s['inputDimension']=s['inputDimension']+1
    
    #LOAD FILES
    mapIndexWord=loadFiles.getWordLocalistMap(wordLocalistMapPath)

    
    corpusLM=CorpusLM()
    corpusLM.loadFromPickle(s['inputFile'])
    
    trainList=corpusLM.training
    testLists=corpusLM.testing
    validateList=corpusLM.validation

    loadFiles.addPeriods(trainList,42)
    for item in trainList:
        loadFiles.addPeriods(item.equivalents,42)
    loadFiles.setInputType(trainList,s['inputType']) 
    
    
    loadFiles.addPeriods(testLists,42)
    for item in testLists:
        loadFiles.addPeriods(item.equivalents,42)
    loadFiles.setInputType(testLists,s['inputType'])
    
    
    loadFiles.addPeriods(validateList,42)
    for item in validateList:
        loadFiles.addPeriods(item.equivalents,42)
    
    loadFiles.setInputType(validateList,s['inputType']) 
    
    validate=validateList

    folderThisRun,bestModelUID,lastModelUID=getFolders(outputsPath,s)
    random.seed(s['seed'])
    
    #CREATE RERANKER SRNN
    srnnRR = rnn.prodSRNN_UID.model(
                              productionModelPath=s['lmProductionFolder'],
                              vocabSize=s['vocab_size'],
                              hiddenDimensProd=s['nhiddenLM'],
                              hiddenDimensRerank= s['nhiddenRR'],
                              dssDimens=    s['inputDimension']
                 )
    
    #IF THE RERANKER HASNT BEEN TRAINED 
    if not s['loadRR']:   
        outputFile= open(folderThisRun+'/outputRR.txt', 'w+')
        best_sim = -numpy.inf
        best_derlen=numpy.inf
        bestEp=0 
        epErrors=[]
        epSimilarities=[]
        s['clr'] = s['lr']
        
        for epoch in xrange(s['nepochs']):
            random.shuffle(trainList)
           
            #TRAIN THIS EPOCH  
            errors=srnnRR.epochTrain(trainList,s['clr']) 
            epErrors.append(sum(errors))
            predictions_validate=srnnRR.getModelPredictions(validate,1)
              
            #Get a list of pairs (sim,mostSimilar) where sim is the similarity of the most similar sentence (mostSimilar) in the gold sentences of the given dss 
            simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(validate,predictions_validate)]
            similarities=[sim for (sim,mostSimilar) in simgolds]
            similarity=numpy.sum(similarities)/len(similarities)    
            epSimilarities.append(similarity)
            
            derLens=[len(predictedSent) for predictedSent in predictions_validate] 
            averageDerLen=numpy.sum(derLens)*1.0/len(derLens)
            
            outputLine='Epoch: '+str(epoch)+' lr: '+str(s['clr'])+" errors: "+str(sum(errors))+' similarity: '+str(similarity)+ " avDerLen: "+str(averageDerLen)
            
            if averageDerLen<best_derlen:
                srnnRR.save(bestModelUID)
                best_derlen = averageDerLen
                bestEp=epoch
                lastChange_LR=epoch#just an aux variable that we can change while keeping track of bestEp
                outputLine='NEW BEST '+outputLine
                
            outputFile.write(outputLine+'\n')
            print outputLine      
                 
            errorsPlot=plt.figure(5000)
            plt.plot(epErrors)
            plt.savefig(folderThisRun+"/errorsTrainEpUID.png")
                          
            simPlot=plt.figure(1000000)
            plt.plot(epSimilarities)
            plt.savefig(folderThisRun+"/similaritiesUID.png")
                 
            # learning rate halves if no improvement in 8 epochs
            if s['decay'] and (epoch-lastChange_LR) >= 8: 
                s['clr'] *= 0.5
                lastChange_LR=epoch#Here we have to reset lastChange_LR, otherwise it will halve each epoch until we get an improvement
                
            #TRAINING STOPS IF THE LEARNING RATE IS BELOW THRESHOLD OR IF NO IMPROVEMENT DURING 30 EPOCHS
            if s['clr'] < 2e-3 or (epoch-bestEp)>=20:     
                break   
             
        srnnRR.save(lastModelUID)
        
        outputLine='BEST RESULT: epoch '+str(bestEp)+' Similarity: '+str(best_sim)+' with the model '+folderThisRun
        print outputLine
        outputFile.write(outputLine)
        outputFile.close()
        
    else:   #IF THE MODEL WAS ALREADY TRAINED AND WE ARE ONLY LOADING IT FOR TESTING
        
        srnnRR.load(lastModelUID)     
           
        outFileTrainUID= open(folderThisRun+'/outputlast_trainUID.txt', 'w+')
        evaluateSRNNUID(srnnRR,outFileTrainUID, validateList,s['periods'])
        outFileTrainUID.close()
        
 #==============================================================================
 #        #Create Language Model
 #        srnnComp = rnn.langmodelSRNN_noBPTT.model(
 #                              inputDimens=s['vocab_size'],
 #                              hiddenDimens =100,
 #                              outputDimens= s['vocab_size'],
 #                              startSentIndex=-1
 #                     )  
 #        srnnComp.load(modelSurprisalPath)
 #           
 #        #GRID SEARCH 
 #        slopeDataP=[]
 #        slopeDataStd=[]
 #        slopeDataAcc=[]
 # 
 #        slopes=numpy.arange(1,5,0.5)
 #        interceps=numpy.arange(1,5,0.5)
 # 
 #        for paramSlope in slopes:
 #            srnnRR.probSlope=paramSlope
 #             
 #            print "SLOPE:"+str(paramSlope)
 #            InterDataP=[]
 #            InterDataStd=[]
 #            InterDataAcc=[]
 #            for paramInterc in interceps:
 #                srnnRR.derLenSlope=paramInterc
 #                 
 #                modelPredictions=srnnRR.getModelPredictions(testLists,1,True)
 #                simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(testLists,modelPredictions)]
 #                similarities=[acc for (acc,gold) in simgolds]
 #                golds=[gold for (acc,gold) in simgolds]
 #         
 #                predWords=[indicesToWords(pred,mapIndexWord) for pred in modelPredictions]
 #                labelWords=[indicesToWords(label,mapIndexWord) for label in golds]
 #                accuracyGlobal=numpy.sum(similarities)/len(similarities)
 #                derlens=[len(pred) for pred in modelPredictions]
 #                print numpy.mean(derlens)
 #                                          
 #                #SURPRISAL VALUES
 #                probabilities=[]
 #                surprisalValues=[]
 #                wordsI=[]
 #                for sent in modelPredictions:
 #                    wordsLocalist=indicesToLocalist(sent,s['vocab_size'])
 #                    sentenceWords=indicesToWords(sent, mapIndexWord)
 #                    _,wordProbs=srnnComp.getSentenceWordProbs(wordsLocalist, sentenceWords)
 #                    probabilities.append(wordProbs)
 #                 
 #                #print surprisals
 #                for sentence in probabilities:
 #                        for (word,wIndex,surpValue) in sentence:
 #                            surprisalValues.append(numpy.log(surpValue)*-1)
 #                            wordsI.append(word)
 # 
 #                InterDataP.append(numpy.mean(surprisalValues))
 #                InterDataStd.append(numpy.std(surprisalValues))
 #                InterDataAcc.append(accuracyGlobal)
 # 
 #            slopeDataP.append(InterDataP)
 #            slopeDataStd.append(InterDataStd)
 #            slopeDataAcc.append(InterDataAcc)
 #         
 #        print slopeDataP
 #        print slopeDataStd
 #        print slopeDataAcc
 #        plt.imshow(slopeDataP, cmap='hot', interpolation='none')
 #        plt.show()
 #        plt.imshow(slopeDataStd,cmap='hot', interpolation='none')
 #        plt.show()
 #        plt.imshow(slopeDataAcc,cmap='hot', interpolation='none')
 #        plt.show()
 #         
 #==============================================================================


        decod=SentenceDecoderUID(srnnRR)
        for item in testLists:
            print
            sentences=decod.getNBestPredictedSentencesPerDSS(item,0.1)
                 
            #SENTENCES AND RANKING ACCORDING TO PRODUCTION MODEL
            sumaModel=0.0
            for sent in sentences:
                sentWords=indicesToWords(sent.indices,mapIndexWord)
                print sentWords
                print str(sent.probability)
                sumaModel+=sent.probability   
                        
            print 
            print "sumProdModelPs:"+str(sumaModel)    
         


     
     
     



