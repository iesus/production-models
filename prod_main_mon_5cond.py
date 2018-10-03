import theano, numpy
import random, os, sys

import matplotlib.pyplot as plt
import data.loadFiles as loadFiles
from data.crossValidation import Fold
from tools.similarities import levenSimilarity
from production.decoder import SentenceDecoder
 
import rnn.prodSRNN_notBPTT_mon as prodSRNN_notBPTT_mon
#import rnn.langmodelSRNN_notBPTT as langmodelSRNN_notBPTT
#import ffnn.ffnn_schemaprobs as ffnn_schemaprobs


sys.path.append("../data")
corpusListsPath="../data/filesSchemas_with150DSS_withSims96/trainTest_Conditions_finalSchemasWithSimilars96_0.pick"
#corpusListsPath="../data/filesSchemas_with150DSS_withSims96/corpusUID.pick"

wordLocalistMapPath='../data/map_localist_words.txt'
dsssMatrixPath="../data/model_vectors"
outputsPath="../outputs"

def xplusplus(name, local={}):
    #Equivalent to name++
    if name in local:
        local[name]+=1
        return local[name]-1
    globals()[name]+=1
    return globals()[name]-1

def mostSimilarEquivalentsLevens(sentenceItem,prediction):
    equivalentsIndices=[localistToIndices(equivalent.wordsLocalist) for equivalent in sentenceItem.equivalents]
    similarities=[levenSimilarity(eq,prediction) for eq in equivalentsIndices]
    mostSimilar=numpy.argmax(similarities, 0)
    
    return (similarities[mostSimilar],equivalentsIndices[mostSimilar])

def localistToIndices(localistMatrix):
    return [numpy.argmax(localist) for localist in localistMatrix]

def indicesToSentence(indices,mapping):
    return [mapping[index] for index in indices]

def getModelPredictions(srnn,testSet,h0,o0,periods):
    predictions_test=[] 
    
    for sentence in testSet:
        predSent=[]
        
        [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h0,o0) #predWord is the index of the highest activated word in the output, h_tm1 is the content of the hidden layer and o_tm1 is the output layer, ie, prob distribution
        predWord=predWord[0] #output is an array of dimension 1
        o_tm1=o0.copy()
        o_tm1[predWord]=1.0
       
        predSent.append(predWord)
     
        if periods:
            while predWord<42 and len(predSent)<20:
                [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h_tm1,o_tm1)
                predWord=predWord[0]
                o_tm1=o0.copy()
                o_tm1[predWord]=1.0
                
                predSent.append(predWord)
        else: 
            for _ in xrange(len(sentence.wordsLocalist)-1):
                [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h_tm1,o_tm1)
                predWord =predWord[0]
                o_tm1=o0.copy()
                o_tm1[predWord]=1.0
                
                predSent.append(predWord)
            
        predictions_test.append(predSent)
    return predictions_test





#Gives sentence probabilities according to a rnn language model trained apart from the production model
def getSentenceProb(srnn_lm, trainingElement,h0):
    wordIndices=localistToIndices(trainingElement.wordsLocalist)
    word0=numpy.zeros(s['vocab_size'], dtype=theano.config.floatX)
    h_tm1=h0
    sentP=1.0
    wordsNonLoc=trainingElement.testItem.split()
    wordsNonLoc.append(".")
    
    for wordLoc,wordIndex,wordNonLoc in zip(trainingElement.wordsLocalist,wordIndices,wordsNonLoc):
        [outProbs,h_tm1,_]=srnn_lm.classify(word0,wordLoc,h_tm1)
        word0=wordLoc
        wordP=outProbs[0][wordIndex]
        sentP=sentP*wordP

    return sentP    


def getSentenceProb_givenSems(prod_srnn, trainingElement,h0):
    wordIndices=localistToIndices(trainingElement.wordsLocalist)
    
    wordInLoc=numpy.zeros(s['vocab_size'], dtype=theano.config.floatX)
    h_tm1=h0
    sentP=1.0
    
    for wordOutLoc,wordIndex in zip(trainingElement.wordsLocalist,wordIndices):
        [_,h_tm1,outProbs]=prod_srnn.classify(trainingElement.input,h_tm1,wordInLoc)
        wordInLoc=wordOutLoc
        wordP=outProbs[wordIndex]
        sentP=sentP*wordP
        print wordP

    return sentP  

def getHiddenActivations(srnn, trainSet, h0,o0):
    activations={}
    counts={}
    
    def getActivatedWords(outLayer):
        activatedItems=[ i for i, word in enumerate(outLayer) if word >0.2]
        return activatedItems
    def addActivation(hiddenLayer,wordIndex):
        if not activations.has_key(wordIndex):
            activations[wordIndex]=hiddenLayer
            counts[wordIndex]=1
        else:
            activations[wordIndex]=activations[wordIndex]+hiddenLayer
            counts[wordIndex]+=1
    
    
    for sentIndex in xrange(len(trainSet)):
        sentence=trainSet[sentIndex]            
        words=sentence.wordsLocalist
        
        [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h0,o0)
        activs=getActivatedWords(o_tm1)
        o_tm1=words[0]
        
        for activWord in activs:
            addActivation(h_tm1,activWord)    
        for i in xrange(len(words)-1):
            word=words[i+1]
            [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h_tm1,o_tm1)
            activs=getActivatedWords(o_tm1)
            o_tm1=word
            for activWord in activs:
                addActivation(h_tm1,activWord)
                
    #print counts       
    for x in xrange(43):
        activations[x]=numpy.divide(activations[x],counts[x])
    return activations,counts
                


def epochTrain(srnn,trainSet,learningRate,h0,o0):
    errors=[]
    for sentIndex in xrange(len(trainSet)):
        sentence=trainSet[sentIndex]            
        words=sentence.wordsLocalist
        
        [e,h_tm1,_]=srnn.train(sentence.input,words[0],learningRate,h0,o0)
        o_tm1=words[0]
       
        errSent=e
        for i in xrange(len(words)-1):
            word=words[i+1]
            [e,h_tm1,_]=srnn.train(sentence.input,word,learningRate,h_tm1,o_tm1)
            o_tm1=word

            errSent+=e
                 
        if sentIndex%25==0:
            errors.append(errSent) 
    return errors

def evaluateSRNN(srnn, outFile, evalSet, h0,o0,periods):
    predictions_test=getModelPredictions(srnn,evalSet,h0,o0,s['periods'])
    
    simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(evalSet,predictions_test)]
    similarities=[acc for (acc,gold) in simgolds]
    golds=[gold for (acc,gold) in simgolds]
       
    predWords=[indicesToSentence(pred,mapIndexWord) for pred in predictions_test]
    labelWords=[indicesToSentence(label,mapIndexWord) for label in golds]
    
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
        
        for (pw,lw,acc) in setValues:
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

def rankingLoss(perfect,proposed):
    weIndex=0
    suma=0
    for (elemPer,elemProp) in zip (perfect,proposed):
        newElem=(elemPer-elemProp)*1.0/(weIndex+1)
        weIndex+=1
        sum+=newElem
    return suma

"""
    Creates the 3 folders where all results/models will be stored
    folderThisRun: folder containing all the files of this particular run, will be contained in folderThisModel which contains all runs of this specific python file
    bestModel: parameters that achieved best performance on the training set
    lastModel: parameters that the model has at the end of training
"""
def getFolders(outputsPath, params):
    #Create folder that contains all the runs for this python file
    currentFolder=outputsPath+"/"+os.path.basename(__file__).split('.')[0]
    folderThisModel=currentFolder+"_outputs"

    if not os.path.exists(folderThisModel): os.mkdir(folderThisModel)
    
    #Create folder for all the files of this specific run
    folderThisRun=folderThisModel+"/output"
    
    folderThisRun+="_"+params['inputType']
    folderThisRun+="_"+str(params['nhidden'])+"h"
    folderThisRun+="_"+str(params['lr'])+"lr"
    folderThisRun+="_"+str(params['nepochs'])+"ep"
    if params['periods']: folderThisRun+="_dots"
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

def wordsToIndices(words,mapping):
    return[mapping[word] for word in words]
    
def getEquivSentencesIndicesSet(trainElem,mapping):
    sentenceWordsSet=[equiv.testItem for equiv in trainElem.equivalents]
    sentenceSplitWordsSet=[sentence.split() for sentence in sentenceWordsSet]  
    return [wordsToIndices(wordsSplitted,mapping) for wordsSplitted in sentenceSplitWordsSet]


if __name__ == '__main__':

    if len(sys.argv)>1:
        x=1
        s={
             'lr':float(sys.argv[xplusplus("x")]), #learning rate
             'verbose':int(sys.argv[xplusplus("x")]),
             'decay':int(sys.argv[xplusplus("x")]), # decay on the learning rate if improvement stops
             'nhidden':int(sys.argv[xplusplus("x")]), # number of hidden units
             'seed':int(sys.argv[xplusplus("x")]),
             
             'percTrain':float(sys.argv[xplusplus("x")]),
             'percValidate':float(sys.argv[xplusplus("x")]),
             'percTest':float(sys.argv[xplusplus("x")]),
             
             'nepochs':int(sys.argv[xplusplus("x")]),
             'label':sys.argv[xplusplus("x")],#"dotsPaper_dssSeparated_811_1",
             'periods':int(sys.argv[xplusplus("x")]),
             'load':int(sys.argv[xplusplus("x")]), #Whether the model is already trained or not
             'inputType':sys.argv[xplusplus("x")], #dss or sitVector or compVector
             'actpas':sys.argv[xplusplus("x")],
             'inputFile':sys.argv[xplusplus("x")] #FILE containing the corpus input
         }
           
    else:
        s = {
         'lr':0.24, #learning rate 0,124
         'verbose':1,
         'decay':True, # decay on the learning rate if improvement stops
         'nhidden':120, # number of hidden units
         'seed':345,
         
         'percTrain':0.9,
         'percValidate':0.0,
         'percTest':0.1,
         
         'nepochs':200,
         #'label':"dotsPaper_dssSeparated_trainstop_normalInitialization_monitor1_4",#"dotsPaper_dssSeparated_811_1",
         'label':"15_40_monitor_sigm_0",#unnormalized2-1-1540
         'periods':True,
         'load':True, #Whether the model is already trained or not
         'inputType':'beliefVector', #dss or sitVector or compVector
         'actpas':True,
         'inputFile':corpusListsPath #FILE containing the corpus input
         }
    if s['periods']: s['vocab_size']=43
    else: s['vocab_size']=42
    
    if s['inputType']=='sitVector' or s['inputType']=='compVector' or s['inputType']=="beliefVector": s['inputDimension']=44
    if s['inputType']=='dss': s['inputDimension']=150
    if s['actpas']:s['inputDimension']=s['inputDimension']+1
    
    #LOAD FILES
    mapIndexWord=loadFiles.getWordLocalistMap(wordLocalistMapPath)
    matrix,events=loadFiles.getAtomicEventDSSMap(dsssMatrixPath)
    
    fold=Fold()
    fold.loadFromPickle(s['inputFile'])
    trainLists=fold.trainSet
    testList=fold.valtestSet

    loadFiles.addPeriods(trainLists[0])
    loadFiles.setInputType(trainLists[0],s['inputType'])
    
    for lista in testList:
        loadFiles.addPeriods(lista)
        for item in lista:
            loadFiles.addPeriods(item.equivalents)
        loadFiles.setInputType(lista,s['inputType'])
    
    train=trainLists[0]
    trainTest=trainLists[1]
    validate=trainTest
    
    folderThisRun,bestModel,lastModel,plotsFolder=getFolders(outputsPath,s)
     
    #CREATE SRNN AND INITIALIZE VARS
    srnn = prodSRNN_notBPTT_mon.model(
                              inputDimens=s['inputDimension'],
                              hiddenDimens = s['nhidden'],
                              outputDimens= s['vocab_size']
                     )        
    
    h0=numpy.zeros(s['nhidden'], dtype=theano.config.floatX)
    o0=numpy.zeros(s['vocab_size'], dtype=theano.config.floatX)
    random.seed(s['seed'])
    
    #IF THE MODEL HASNT BEEN TRAINED 
    if not s['load']:   
        outputFile= open(folderThisRun+'/output.txt', 'w+')
        best_sim = -numpy.inf
        bestEp=0
        epErrors=[]
        epSimilarities=[]         
        s['clr'] = s['lr']
        
        for epoch in xrange(s['nepochs']):
            random.shuffle(train)
                 
            #TRAIN THIS EPOCH
            errors=epochTrain(srnn,train,s['clr'],h0,o0)
            epErrors.append(sum(errors))
            
            #HERE IS THE DIFFERENCE BETWEEN TRAINING NUMBERS OF THE TRAIN SET AND TESTING NUMBERS ALSO OF THE TRAIN SET (the periods flag)
            predictions_validate=getModelPredictions(srnn,validate,h0,o0,0) #We dont stop on periods, because at the beginning the system may not know that it has to put a period
             
            #Get a list of pairs (sim,mostSimilar) where sim is the similarity of the most similar sentence (mostSimilar) in the gold sentences of the given dss 
            simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(validate,predictions_validate)]
            #Get only the list of similarities
            similarities=[sim for (sim,mostSimilar) in simgolds]
            similarity=numpy.sum(similarities)/len(similarities)    
            epSimilarities.append(similarity)    
            
            if similarity > best_sim:
                srnn.save(bestModel)
                best_sim = similarity
                bestEp=epoch
                lastGood=epoch#just an aux variable that we can change while keeping track of bestEp
                        
                outputFile.write('NEW BEST EPOCH: '+str(epoch)+' similarity: '+str(similarity)+"\n")
                print 'NEW BEST: epoch', epoch, 'similarity', similarity
            
            else:
                outputFile.write('Epoch: '+str(epoch)+' lr: '+str(s['clr'])+' similarity: '+str(similarity)+'\n')
                print 'Epoch:',epoch,'lr:',s['clr'],'similarity:',similarity      
                 
            errorsPlot=plt.figure(100000)
            plt.plot(epErrors)
            plt.savefig(folderThisRun+"/errorsTrainEp.png")
                          
            simPlot=plt.figure(1000000)
            plt.plot(epSimilarities)
            plt.savefig(folderThisRun+"/similarities.png")
                
            # learning rate halves if no improvement in 15 epochs
            if s['decay'] and (epoch-lastGood) >= 15: 
                s['clr'] *= 0.5
                lastGood=epoch#Here we have to reset lastGood, otherwise it will halve each epoch until we get an improvement
                
            #TRAINING STOPS IF THE LEARNING RATE IS BELOW THRESHOLD OR IF NO IMPROVEMENT DURING 40 EPOCHS
            if s['clr'] < 1e-3 or (epoch-bestEp)>=40:     
                break  
            
        srnn.save(lastModel)
        
        print 'BEST RESULT: epoch', bestEp, 'Similarity:', best_sim,'with the model', folderThisRun
        outputFile.write('BEST RESULT: epoch '+str(bestEp)+' Similarity: '+str(best_sim)+' with the model '+folderThisRun)
        outputFile.close()
      
    else: #IF THE MODEL WAS ALREADY TRAINED AND WE ARE ONLY LOADING IT FOR TESTING

        srnn.load(lastModel)
        trainingSents=[item.testItem for item in trainLists[0]]
        
        random.shuffle(trainingSents)
        
        for sent in trainingSents[:100]:
            print "\item "+sent
        
        
       #========================================================================
       #  class WordContinuationsDict:
       #      def __init__(self,word,otherWords):
       #          self.word=word
       #          self.wordContinuationsCounts={word:0 for word in otherWords}
       #  
       #  listOfWords=[word for word in mapIndexWord.itervalues()]
       #  
       #  allCounts={}
       #  allProbs={}
       #  for word in listOfWords:
       #      contDict=WordContinuationsDict(word,listOfWords)
       #      probDict=WordContinuationsDict(word,listOfWords)
       #      allCounts[word]=contDict
       #      allProbs[word]=probDict
       # 
       #  
       #  for sentence in trainingSents:
       #      wordsSentence= sentence.split()
       #      previousWord=0
       #      for word in wordsSentence:
       #          if previousWord:
       #              allCounts[previousWord].wordContinuationsCounts[word]+=1
       #          previousWord=word
       #  
       #  for key,value in allCounts.iteritems():
       #      #print key,value.wordContinuationsCounts,sum(value.wordContinuationsCounts.values())
       #      totalCounts=sum(value.wordContinuationsCounts.values())*1.0
       #      if totalCounts:
       #          allProbs[key].wordContinuationsCounts={k: v / totalCounts for k, v in value.wordContinuationsCounts.iteritems()}
       #      else:
       #          allProbs[key].wordContinuationsCounts=value.wordContinuationsCounts
       #      
       #      #print key,allProbs[key].wordContinuationsCounts    
       #   
       #  matrixProbs=[]
       #  for word in listOfWords:
       #      rowProb=[allProbs[word].wordContinuationsCounts[word1] for word1 in listOfWords]
       #      #rowProb=[prob for prob in allProbs[word].wordContinuationsCounts.itervalues()]
       #      matrixProbs.append(rowProb)
       #========================================================================
            
         
        #import plotly
        
        def network_analysis_part():
            import network_analysis
            activs,counts=getHiddenActivations(srnn,train,h0,o0)
            
            inputW=srnn.W_xh.eval()
            outputW=srnn.W_hy.eval()
            contextW=srnn.W_hh.eval()
            
            mapIndexWord[18]="hide&seek" #instead of hide_and_seek 
            
            wordInfos,relevanceHidden=network_analysis.getHiddenRelevance(srnn,activs,mapIndexWord,normalization=True)
            relevanceHiddenMatrix=numpy.asmatrix(relevanceHidden) 
            #mapAHidWords,mapIHidWords=network_analysis.getHiddenUnitWords(wordInfos)                
    
            #mapRelHidWords=network_analysis.getRelHidWords(wordInfos) #Get a map from each hidden unit to its relevance values of the output
            #mapActHidWords,mapInhHidWords=network_analysis.separateActInhRelHidWords(mapRelHidWords)#separate the map into activation and inhibition
    
            
            printList=[0,35,9,17,33,7,16,32,10,18,14,31,3,12,22,29,15,37,4,30,6,27,34,20,25,5,21,23,28,39,24,26,41,2,38,11,13,1,8,19,36,40,42]
            wordList=[mapIndexWord[x] for x in printList]
            
            #plotly.tools.set_credentials_file(username='jesusct2', api_key='VoDVZmLfN22kJCln3bCT')
            plotly.tools.set_credentials_file(username='jesusct', api_key='K0L2vwH3cCZAs1LjdCpQ')        
    
            bwr=network_analysis.getCMapForPlotly("bwr")
            #seismic=network_analysis.getCMapForPlotly("seismic")
            
            #import plotly.plotly as py
            #import plotly.graph_objs as go
            
            #HIDDEN UNITS RELEVANCE!!
            #selectedHUnits=[0,1,2,3,4,10,30,34,35,36,69,80,111,115]
            #network_analysis.createHeatmapHiddenUnits(mapRelHidWords,selectedHUnits,wordList,printList,filename="selectedHUnits",colormap=bwr,minV=-0.11,maxV=0.11,title="Selected Hidden Units1")
    
            #MONITORING RELEVANCE!!!!
            monitorW=srnn.W_oh.eval()
            network_analysis.createHeatmapMonitorUnits(monitorW,relevanceHiddenMatrix,bwr,printList,wordList,True,filename="monitorHeatmapAct",title="Monitoring Units Activation",height=1000,width=900,offline=False)
            network_analysis.createHeatmapProbs(matrixProbs,bwr,printList,wordList,True,filename="probsHeatmap",title="Monitoring Probs",height=1000,width=900,offline=False)
        #network_analysis_part()
            
            #INPUUUUUT STUFF RELEVANCE!!!!  
#             inputUnitsLabels=["play(charlie,chess)","play(charlie,hide&seek)","play(charlie,soccer)","play(heidi,chess)","play(heidi,hide&seek)","play(heidi,soccer)",
#                         "play(sophia,chess)","play(sophia,hide&seek)","play(sophia,soccer)","play(charlie,puzzle)","play(charlie,ball)","play(charlie,doll)",
#                         "play(heidi,puzzle)","play(heidi,ball)","play(heidi,doll)","play(sophia,puzzle)","play(sophia,ball)","play(sophia,doll)",
#                         "win(charlie)","win(heidi)","win(sophia)","lose(charlie)","lose(heidi)","lose(sophia)","place(charlie,bathroom)","place(charlie,bedroom)",
#                         "place(charlie,playground)","place(charlie,street)","place(heidi,bathroom)","place(heidi,bedroom)","place(heidi,playground)","place(heidi,street)",
#                         "place(sophia,bathroom)","place(sophia,bedroom)","place(sophia,playground)","place(sophia,street)","manner(play(charlie),well)","manner(play(charlie),badly)",
#                         "manner(play(heidi),well)","manner(play(heidi),badly)","manner(play(sophia),well)","manner(play(sophia),badly)","manner(win,easily)","manner(win,difficultly)","actives"]
#             
            #network_analysis.createHeatmapInputUnits(inputW,relevanceHiddenMatrix,bwr,printList,wordList,inputUnitsLabels,normalization=False,filename="testinput",minV=-2,maxV=2,title="Input Units",height=1200,width=1000,offline=True)
                   
            #CONTEXT RELEVANCE!!!!
            #network_analysis.createHeatmapContextUnits(contextW,relevanceHiddenMatrix,bwr,printList,wordList,normalizationCon=True,filename="contextHeatmapALL",minV=-0.4, maxV=0.4,title="Context Units",height=1000,width=900,offline=True)
            
            #TIME STEP 0
            #Gets the 10 most positive and negative weights of input->hidden and shows the words related to those hidden units
            #network_analysis.getActivationsInhibitionsOf10LargestInputWeights(inputW,mapAHidWords,mapIHidWords,inputUnitsLabels)
            
            #posM,negM=network_analysis.separatePositiveNegativeMatrix(inputW)
            #mapA,mapI=network_analysis.getTotalActivationInhibitionPerWord_OnlyMostPerOutput(inputW,mapAHidWords,mapIHidWords)
            #network_analysis.sumOutputActivationsInhibitions(outputW,mapIndexWord)
            
    
#             #RANK WORDS ACCORDING TO PROBABILITY OF OCCURRING AT THE BEGINNING OF A SENTENCE        
#             def inputPos1WordRank():
#                 mapIndexWord[18]="hide_and_seek"
#                 mapWordIndex={word:index for index,word in mapIndexWord.items()}
#                
#                 #Initialize dictionaries 
#                 pos1DictCounter={}
#                 for (key,value) in mapIndexWord.items():
#                     pos1DictCounter[value]=0
#                 
#                 for trainElem in train:    
#                     sentWords=trainElem.testItem.split()        
#                     pos1DictCounter[sentWords[0]]+=1
#                 
#                 tuplesPos1=pos1DictCounter.items()        
#                 import operator
#                 tuplesPos1.sort(key=operator.itemgetter(1), reverse=True)
#                 
#                 rankPos1=[mapWordIndex[word] for (word,val) in tuplesPos1]
#                 wordListPos1=[word for (word,val) in tuplesPos1]
#                 
#                 hideseekInd=wordListPos1.index("hide_and_seek")
#                 wordListPos1[hideseekInd]="hide&seek"
#                 
#                         
#                 def mergeRanks(rankList1,rankList2,mergePoint):
#                     finalRank=rankList1[:mergePoint]
#                     for elem in rankList2:
#                         if elem not in finalRank:
#                             finalRank.append(elem) 
#                     return finalRank
#                     
#                 rankPos1=mergeRanks(rankPos1,printList,10)
#                 wordListPos1=[mapIndexWord[i] for i in rankPos1]
#             
#                 #original Ranking
#                 #network_analysis.createHeatmapInputUnits(inputW,relevanceHiddenMatrix,bwr,printList,wordList,inputUnitsLabels,normalization=False,filename="non-normalizedInputRelevance",minV=-2,maxV=2,title="Input Units Non-normalized",height=1200,width=1000,offline=False)    
#                 #first the ones possible at t0
#                 network_analysis.createHeatmapInputUnits(inputW,relevanceHiddenMatrix,bwr,rankPos1,wordListPos1,inputUnitsLabels,normalization=False,filename="rankPos1T0",minV=-2,maxV=2,title="Input Units RankPos1 T0",height=1200,width=1000,offline=False)
#                 #Results are not very nice of the following line
#                 #network_analysis.createHeatmapInputUnits(inputW,relevanceHiddenMatrix,bwr,rankPosAver,wordListPosAver,inputUnitsLabels,normalization=False,filename="rankPosAverT0",minV=-2,maxV=2,title="Input Units RankPosAver T0",height=1200,width=1000,offline=True)

#         #NORMAL TESTING
#         outFileTrain= open(folderThisRun+'/outputlast_train.txt', 'w+')
#         outFileTest= open(folderThisRun+'/outputlast_test.txt', 'w+')
#              
#              
#         #=======================================================================
#         # oneItem = trainTest[0]
#         # print oneItem.testItem
#         # print getSentenceProb_givenSems(srnn, oneItem,h0)
#         # 
#         #=======================================================================
#         for tItem in trainTest[:10]:
#             print tItem.testItem
#             print getSentenceProb_givenSems(srnn,tItem,h0)
#             print
#             
#         
#         
#         
#         exit()
#         evaluateSRNN(srnn, outFileTrain, trainTest, h0, o0, s['periods'])
#         outFileTrain.close()
#         
#         for index in xrange(len(testList)):
#             print "\nCONDITION:"+str(index+1)+"\n"
#             outFileTest.write("\nCONDITION:"+str(index+1)+"\n")
#             evaluateSRNN(srnn, outFileTest, testList[index],  h0, o0, s['periods'])
#            
#         outFileTest.close()
         
         
#HERE ENDS NORMAL TESTING
        




#===============================================================================
#     #ALL CONDITIONS ALL FOLDS TESTING
#         import data.setAnalizer as setAnalyzer  
#         mapWordIndex={word:index for index,word in mapIndexWord.items()}    
#         foldsValues=[[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]
#         goodsSetSizes=[]
#         badsSetSizes=[]
#         
#         for x in xrange(10):
#             modelPath= "/Users/jesus/Documents/PapersPropios/JournalPaper/outputs/prod_main_mon_5cond_outputs/output_beliefVector_120h_0.24lr_200ep_dots_5cond_25K_"+str(x)+"/lastModel"
#             foldPath="/Users/jesus/Documents/EclipseWorkspace/productionModel/data/serverData2/filesSchemas_with150DSS_withSims96/trainTest_Conditions_finalSchemasWithSimilars96_"+str(x)+".pick"
#             
#             fold=Fold()
#             fold.loadFromPickle(foldPath)
#             trainLists=fold.trainSet
#             testList=fold.valtestSet
#         
#             loadFiles.addPeriods(trainLists[0])
#             loadFiles.setInputType(trainLists[0],s['inputType'])
#             
#             for lista in testList:
#                 loadFiles.addPeriods(lista)
#                 for item in lista:
#                     loadFiles.addPeriods(item.equivalents)
#                 loadFiles.setInputType(lista,s['inputType'])
#             
#             train=trainLists[0]
#             trainTest=trainLists[1]
#             validate=trainTest
#             
#             
#             
#             srnn.load(modelPath)
#             decod=SentenceDecoder(srnn,h0,o0,mapIndexWord) 
#             listasCond=[testList[0],testList[1],testList[2],testList[3],testList[5]]
#             
#             for index in xrange(len(listasCond)): 
#                 lista=listasCond[index]
#                 allValues=[[],[],[]]
#                 
#                 bads=0  #goods are 70-bad
#                 for item in lista:
#                     indicesSet=getEquivSentencesIndicesSet(item,mapWordIndex) 
#                     sentencesModel=decod.getNBestPredictedSentencesPerDSS(item,0.12,"noSentencePlots",False,None)
#                     sentencesModelIndices=[sent.indices for sent in sentencesModel]
#         
#                     prec,rec,fscore=setAnalyzer.precisionRecallFScore(indicesSet,sentencesModelIndices)
#                     if fscore<1.0:
#                         bads+=1
#                         badsSetSizes.append(len(item.equivalents))
#                     else:
#                         goodsSetSizes.append(len(item.equivalents))
#                     
#                     allValues[0].append(prec)
#                     allValues[1].append(rec)
#                     allValues[2].append(fscore)
#                     
#                 avPrec=numpy.mean(allValues[0], axis=0)
#                 avRec=numpy.mean(allValues[1], axis=0)
#                 avFsc=numpy.mean(allValues[2], axis=0)
#                 
#                 foldsValues[0][index].append(avPrec)
#                 foldsValues[1][index].append(avRec)
#                 foldsValues[2][index].append(avFsc)
#                 foldsValues[3][index].append(14-bads)
#             
#                 
#         print foldsValues
#         condPrec= numpy.mean(foldsValues[0],axis=1)
#         condRec= numpy.mean(foldsValues[1],axis=1)
#         condFSc= numpy.mean(foldsValues[2],axis=1)
#         condPerf=numpy.mean(foldsValues[3],axis=1)/14.0
#         
#         avPrecA=numpy.mean(condPrec)
#         avRecA=numpy.mean(condRec)
#         avFscA=numpy.mean(condFSc)
#         avPerf=numpy.mean(condPerf)
#         
#         avGoodSetSize=numpy.mean(goodsSetSizes)
#         sdGoodSetSize=numpy.std(goodsSetSizes)
#         avBadSetSize=numpy.mean(badsSetSizes)
#         sdBadSetSize=numpy.std(badsSetSizes)
#         
#         #=======================================================================
#         # avPrecA=numpy.mean(numpy.take(condPrec,[0,1,2,3,5]))
#         # avRecA=numpy.mean(numpy.take(condRec,[0,1,2,3,5]))
#         # avFscA=numpy.mean(numpy.take(condFSc,[0,1,2,3,5]))
#         #=======================================================================
#         
#         print condPrec
#         print avPrecA
# 
#         print condRec
#         print avRecA
#         
#         print condFSc
#         print avFscA
#         
#         print condPerf
#         print avPerf
#         
#         print goodsSetSizes
#         print avGoodSetSize
#         print sdGoodSetSize
#         print
#         print badsSetSizes
#         print avBadSetSize
#         print sdBadSetSize
#===============================================================================
        #=======================================================================
        # verybad=False
        # verybads=[]
        # #TESTING WITH DEEPER ANALYSIS
        # mapWordIndex={word:index for index,word in mapIndexWord.items()}
        #  
        # import data.setAnalizer as setAnalyzer
        # from data.derivationTreeDSS import DerivationTree 
        # from data.derivationTreeDSS import SimpleDerivationTree, TreeComparer 
        # from collections import Counter
        # boolPlots=False
        # overgenerationsAll=Counter({})
        # undergenerationsAll=Counter({})
        #  
        # countthis=0
        # countgood=0
        # countbad=0
        # sitsAGame=[]
        # 
        # outFile= open('/Users/jesus/Desktop/outputPassives.txt', 'w+')
        #  
        #  
        # for x in xrange(3):
        #     outFile.write("NEW FOLD\n")
        #     modelPath= "/Users/jesus/Documents/PapersPropios/JournalPaper/outputs/prod_main_mon_5cond_outputs/output_beliefVector_120h_0.24lr_200ep_dots_5cond_25K_"+str(x)+"/lastModel"
        #     foldPath="/Users/jesus/Documents/EclipseWorkspace/productionModel/data/serverData2/filesSchemas_with150DSS_withSims96/trainTest_Conditions_finalSchemasWithSimilars96_"+str(x)+".pick"
        #       
        #     fold=Fold()
        #     fold.loadFromPickle(foldPath)
        #     trainLists=fold.trainSet
        #     testList=fold.valtestSet
        #   
        #     loadFiles.addPeriods(trainLists[0])
        #     loadFiles.setInputType(trainLists[0],s['inputType'])
        #       
        #     for lista in testList:
        #         loadFiles.addPeriods(lista)
        #         for item in lista:
        #             loadFiles.addPeriods(item.equivalents)
        #         loadFiles.setInputType(lista,s['inputType'])
        #       
        #     train=trainLists[0]
        #     trainTest=trainLists[1]
        #     validate=trainTest
        #       
        #       
        #     srnn.load(modelPath)
        #     decod=SentenceDecoder(srnn,h0,o0,mapIndexWord) 
        #      
        #     overgenerationsFold=Counter({})
        #     undergenerationsFold=Counter({})
        #     comparer=TreeComparer()
        #     
        #     testconds=[testList[0],testList[1],testList[2],testList[3],testList[5]]
        #     testPassives=[testList[4],testList[6]]
        #  
        #     for lista in testPassives:
        #         outFile.write("NEW LIST:\n")
        #         interestSits=[]
        #         for item in lista:
        #               
        #             #indicesSet=getEquivSentencesIndicesSet(item,mapWordIndex) 
        #               
        #             #tree=DerivationTree(item)
        #             #tree.processDerLengthsInfo(mapWordIndex)
        #             #treeRoot=tree.root
        #             #sentsExpected=[equiv.testItem for equiv in item.equivalents]
        #             #tree1=SimpleDerivationTree(sentsExpected)
        #             
        #             sentences=decod.getNBestPredictedSentencesPerDSS(item,0.12,plotsFolder,boolPlots)
        #             
        #             sentencesModelIndices=[sent.indices for sent in sentences] 
        #             sentencesWordsSplit=[indicesToSentence(sent.indices,mapIndexWord) for sent in sentences]
        #             sentencesStringList=[" ".join(splitSent) for splitSent in sentencesWordsSplit]
        #             
        #             print item.testItem
        #             outFile.write("\n"+item.testItem+"\n")
        #             for sent in sentencesStringList:
        #                 print "\t"+sent
        #                 outFile.write("\t"+sent+"\n")
        #             print
        #             
        # outFile.close()
        #=======================================================================
        #=======================================================================
        #             sentencesModelIndices=[sent.indices for sent in sentences] 
        #             sentencesWordsSplit=[indicesToSentence(sent.indices,mapIndexWord) for sent in sentences]
        #             sentencesStringList=[" ".join(splitSent) for splitSent in sentencesWordsSplit]
        #             tree2=SimpleDerivationTree(sentencesStringList)
        #             
        #             comparer.nodeCompare(tree1.root, tree2.root)
        #             overgenerationsFold=comparer.countOver+overgenerationsFold
        #             undergenerationsFold=comparer.countUnder+undergenerationsFold
        #             if comparer.countUnder.has_key("at"):sitsAGame.append((item.equivalents,sentencesStringList))
        #                               
        #             prec,rec,fscore=setAnalyzer.precisionRecallFScore(indicesSet,sentencesModelIndices)
        #               
        #             if fscore<1.0:
        #                     print
        #                     print comparer.countOver
        #                     print comparer.countUnder
        #                     interestSits.append((sentences,indicesSet,item))
        #                     print "PROBLEMATIC STUFFFF!!!"
        #                     print "Expected:"
        #                     for testIt in item.equivalents:
        #                         print testIt.testItem
        #                     print len(item.equivalents)
        #       
        #                     print
        #                     print item.hasSimilar
        #                     sumaModel=0.0
        #                     for sent,sentWords in zip(sentences,sentencesStringList):
        #                         print str(sent.probability)+"    \t"+str(sentWords)
        #                         sumaModel+=sent.probability   
        #                     print len(sentences)
        #                     print "sumProdModelPs:"+str(sumaModel)  
        #                     countbad+=1
        #                     
        #                     if len(comparer.countUnder.keys())>0:
        #                         countthis+=1
        #                         outFile.write("\n")
        #                         for sentex in sentsExpected:
        #                             outFile.write(sentex+"\n")
        #                         outFile.write("\n")
        #                         for gener in sentencesStringList:
        #                             if gener not in sentsExpected:
        #                                 outFile.write("\tover:\t"+gener+"\n")
        #                         for expec in sentsExpected:
        #                             if expec not in sentencesStringList:
        #                                 outFile.write("\t\tunder:\t"+expec+"\n")
        #                     
        #             
        #             else: countgood+=1
        #             if rec==0.0:
        #                 veryBad=True
        #                 verybads.append((item.equivalents,sentencesStringList))
        #                 
        #             comparer.flush()
        #     print overgenerationsFold
        #     print undergenerationsFold    
        #     overgenerationsAll=overgenerationsAll+overgenerationsFold
        #     undergenerationsAll=undergenerationsAll+undergenerationsFold
        #      
        # print overgenerationsAll
        # print undergenerationsAll
        # print countgood
        # print countbad
        # print countthis
        # print veryBad
        # 
        # print len(verybads)
        # outFile.close()
        #=======================================================================
        #=======================================================================
        # for (expected,sents) in verybads:
        #     for equiv in expected:
        #         print equiv.testItem
        #     print "OUTPUT:"
        #     for senty in sents:
        #         print senty
        #     print
        # 
        # print "SITS WITH A GAME"
        # for (sentexp,sentgiven) in sitsAGame:
        #     for sente in sentexp:
        #         print sente.testItem
        #     print "OUTPUT"
        #     for sentg in sentgiven:
        #         print sentg
        #     print
        #=======================================================================
