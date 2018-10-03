'''
Created on May 6, 2016
'''
import theano, numpy
import random
import os,sys

import matplotlib.pyplot as plt
import data.loadFiles as loadFiles
from data.crossValidation import Fold
from data.containers import CorpusUID
from tools.similarities import levenSimilarity
from production.decoder import SentenceDecoder, SentenceDecoderUID
 
import rnn.prodSRNN_notBPTT_mon as prodSRNN_notBPTT_mon
import rnn.prodSRNN_notBPTT_UID as prodSRNN_notBPTT_UID
import rnn.langmodelSRNN_notBPTT as langmodelSRNN_notBPTT

sys.path.append("../data")
corpusListsPath="../data/filesSchemas_with150DSS_withSims96/corpusUID.pick"

wordLocalistMapPath='../data/map_localist_words.txt'
dsssMatrixPath="../data/model_vectors"
outputsPath="../outputs"

modelProductionPath="/Users/jesus/Documents/EclipseWorkspace/productionModel/outputs/prod_main_prob_UID_outputs/output_beliefVector_120h_0.24lr_200ep_dots_15_40_monitor_sigm_0/lastModel"
modelSurprisalPath="/Users/jesus/Documents/EclipseWorkspace/productionModel/outputs/langModel_UIDPaper_outputs/output_beliefVector_120h_0.24lr_200ep_dots_15_40_monitor_sigm_0/lastModel"
#modelPath= "/Users/jesus/Documents/PapersPropios/JournalPaper/outputs/prod_main_mon_5cond_outputs/output_beliefVector_120h_0.24lr_200ep_dots_5cond_25K_0/lastModel"
#corpusListsPath="/Users/jesus/Documents/EclipseWorkspace/productionModel/data/serverData2/filesSchemas_with150DSS_withSims96/trainTest_Conditions_finalSchemasWithSimilars96_0.pick"

def postIncre(name, local={}):
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
    for sentIndex in xrange(len(testSet)):
        sentence=testSet[sentIndex]            
    
        predSent=[]
        [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h0,o0)
        
        indexW = numpy.argmax(o_tm1)
        o_tm1=o0.copy()
        o_tm1[indexW]=1.0
        
        predSent.append(predWord[0])
     
        if periods:
            while predWord[0]<42 and len(predSent)<20:
                [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h_tm1,o_tm1)
                indexW = numpy.argmax(o_tm1)
                o_tm1=o0.copy()
                o_tm1[indexW]=1.0
                
                predSent.append(predWord[0])
        else: 
            for _ in xrange(len(sentence.wordsLocalist)-1):
                [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h_tm1,o_tm1)
                indexW = numpy.argmax(o_tm1)
                o_tm1=o0.copy()
                o_tm1[indexW]=1.0
                
                predSent.append(predWord[0])
            
        predictions_test.append(predSent)
    return predictions_test

def getModelPredictionsUID(srnnLM,srnnRR,testSet,h0,o0,h0R,periods):
    predictions_test=[] 
    for sentIndex in xrange(len(testSet)):
        sentence=testSet[sentIndex]            
        p_dss=len(sentence.equivalents)*1.0/130
        dl_f=p_dss-2
        pr_f=1-p_dss+2
        #print p_dss
        predSent=[]
        [_,h_tm1,o_tm1]=srnnLM.classify(sentence.input,h0,o0)
        #=======================================================================
        # print "antes"
        # print o_tm1
        # print numpy.argmax(o_tm1)
        # for indexa in xrange(len(o_tm1)):
        #     if o_tm1[indexa]<0.12:o_tm1[indexa]=0
        # print o_tm1
        # print numpy.argmax(o_tm1)
        # print "despues"
        #=======================================================================
        [h_tm1R,predOutDistro]=srnnRR.classify(sentence.input,o_tm1,h0R)
        multi=dl_f*predOutDistro+o_tm1*pr_f
        #indexW = numpy.argmax(predOutDistro)
        #indexW=numpy.argmax(o_tm1)
        indexW=numpy.argmax(multi)
        o_tm1=o0.copy()
        o_tm1[indexW]=1.0
        
        predSent.append(indexW)
        
        if periods:
            while indexW<42 and len(predSent)<20:
                [_,h_tm1,o_tm1]=srnnLM.classify(sentence.input,h_tm1,o_tm1)
                #===============================================================
                # for index in xrange(len(o_tm1)):
                #     if o_tm1[index]<0.12:
                #         o_tm1[index]=0
                #===============================================================
                [h_tm1R,predOutDistro]=srnnRR.classify(sentence.input,o_tm1,h_tm1R)
                multi=dl_f*predOutDistro+o_tm1*pr_f
                #===============================================================
                # print "lengts"
                # print predOutDistro
                # print "probs"
                # print o_tm1
                # print "both"
                # print multi
                #===============================================================
                indexW=numpy.argmax(multi)
                #indexW = numpy.argmax(predOutDistro)
                #indexW=numpy.argmax(o_tm1)
                o_tm1=o0.copy()
                o_tm1[indexW]=1.0
                
                predSent.append(indexW)
        else: 
            for _ in xrange(len(sentence.wordsLocalist)-1):
                [_,h_tm1,o_tm1]=srnnLM.classify(sentence.input,h_tm1,o_tm1)
                #===============================================================
                # for index in xrange(len(o_tm1)):
                #     if o_tm1[index]<0.12:
                #         o_tm1[index]=0
                #===============================================================
                [h_tm1R,predOutDistro]=srnnRR.classify(sentence.input,o_tm1,h_tm1R)
                indexW = numpy.argmax(predOutDistro)
                o_tm1=o0.copy()
                o_tm1[indexW]=1.0
                
                predSent.append(indexW)
            
        predictions_test.append(predSent)
    return predictions_test

def getModelPredictionsUIDTest(srnnLM,srnnRR,testSet,h0,o0,h0R,periods,paramI,paramS):
    predictions_test=[] 
    for sentIndex in xrange(len(testSet)):
        sentence=testSet[sentIndex]            
        p_dss=len(sentence.equivalents)*1.0/paramS
        #dl_f=p_dss-paramI
        dl_f=-paramI
        #pr_f=1-p_dss+paramI
        pr_f=paramS
        #print p_dss
        predSent=[]
        [_,h_tm1,o_tm1]=srnnLM.classify(sentence.input,h0,o0)
        #=======================================================================
        # print "antes"
        # print o_tm1
        # print numpy.argmax(o_tm1)
        # for indexa in xrange(len(o_tm1)):
        #     if o_tm1[indexa]<0.12:o_tm1[indexa]=0
        # print o_tm1
        # print numpy.argmax(o_tm1)
        # print "despues"
        #=======================================================================
        [h_tm1R,predOutDistro]=srnnRR.classify(sentence.input,o_tm1,h0R)
        multi=dl_f*predOutDistro+o_tm1*pr_f
        #indexW = numpy.argmax(predOutDistro)
        #indexW=numpy.argmax(o_tm1)
        indexW=numpy.argmax(multi)
        o_tm1=o0.copy()
        o_tm1[indexW]=1.0
        
        predSent.append(indexW)
        
        if periods:
            while indexW<42 and len(predSent)<20:
                [_,h_tm1,o_tm1]=srnnLM.classify(sentence.input,h_tm1,o_tm1)
                #===============================================================
                # for index in xrange(len(o_tm1)):
                #     if o_tm1[index]<0.12:
                #         o_tm1[index]=0
                #===============================================================
                [h_tm1R,predOutDistro]=srnnRR.classify(sentence.input,o_tm1,h_tm1R)
                multi=dl_f*predOutDistro+o_tm1*pr_f
                #===============================================================
                # print "lengts"
                # print predOutDistro
                # print "probs"
                # print o_tm1
                # print "both"
                # print multi
                #===============================================================
                indexW=numpy.argmax(multi)
                #indexW = numpy.argmax(predOutDistro)
                #indexW=numpy.argmax(o_tm1)
                o_tm1=o0.copy()
                o_tm1[indexW]=1.0
                
                predSent.append(indexW)
        else: 
            for _ in xrange(len(sentence.wordsLocalist)-1):
                [_,h_tm1,o_tm1]=srnnLM.classify(sentence.input,h_tm1,o_tm1)
                #===============================================================
                # for index in xrange(len(o_tm1)):
                #     if o_tm1[index]<0.12:
                #         o_tm1[index]=0
                #===============================================================
                [h_tm1R,predOutDistro]=srnnRR.classify(sentence.input,o_tm1,h_tm1R)
                indexW = numpy.argmax(predOutDistro)
                o_tm1=o0.copy()
                o_tm1[indexW]=1.0
                
                predSent.append(indexW)
            
        predictions_test.append(predSent)
    return predictions_test

#THIS METHOD IS VERY PROBABLY WRONG, CHECK getSentenceProbs
def getSentenceProb(srnn, trainingElement,h0,w0):
    wordIndices=localistToIndices(trainingElement.wordsLocalist)
    
    h_tm1=h0
    sentP=1.0
    wordsNonLoc=trainingElement.testItem.split()
    wordsNonLoc.append(".")
    word0=w0
    for wordLoc,wordIndex,wordNonLoc in zip(trainingElement.wordsLocalist,wordIndices,wordsNonLoc):
        [outProbs,h_tm1,_]=srnn.classify(word0,wordLoc,h_tm1)
        word0=wordLoc
        wordP=outProbs[0][wordIndex]
        sentP=sentP*wordP

    return sentP    

def getSentenceProbs(srnnComp,wordIndices,h0,w0):
    sentenceWords=indicesToSentence(wordIndices,mapIndexWord)
    sentenceProbs=[]
    
    h_tm1=h0
    word0=w0
    
    wordLoc=w0.copy()
    wordLoc[wordIndices[0]]=1
    [outProbs,h_tm1,_]=srnnComp.classify(word0,wordLoc,h_tm1)
    wordP=outProbs[0][wordIndices[0]]
    sentenceProbs.append((sentenceWords[0],wordIndices[0],wordP))    
    
    for x in xrange(1,len(wordIndices)):
        word0=wordLoc
        wordLoc=w0.copy()
        wordLoc[wordIndices[x]]=1
        [outProbs,h_tm1,_]=srnnComp.classify(word0,wordLoc,h_tm1)
        
        wordP=outProbs[0][wordIndices[x]]
        sentenceProbs.append((sentenceWords[x],wordIndices[x],wordP))

    return sentenceProbs


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

def epochTrainRR(srnnLM,srnnRR,trainSet,learningRate,h0,o0,h0R):
    errors=[]
    for sentIndex in xrange(len(trainSet)):
        item=trainSet[sentIndex]
        words=item.wordsLocalist      
         
        expectedVectors=numpy.asarray(item.lengthVectors)
        #=======================================================================
        # print expectedVectors[0]
        # print item.input
        # print h0
        # print o0
        #=======================================================================
        [wordMaxP,h_tm1LM,o_tm1]=srnnLM.classify(item.input,h0,o0)
        #=======================================================================
        # print "YEAH"
        # print wordMaxP[0]
        # print o_tm1
        # print learningRate
        # print h0R
        #=======================================================================
        #=======================================================================
        # for index in xrange(len(o_tm1)):
        #     if o_tm1[index]<0.12:
        #         o_tm1[index]=0
        #=======================================================================
        #print o_tm1
        #print "yeah2"
        [e,h_tm1RR,_]=srnnRR.train(item.input,o_tm1,expectedVectors[0],learningRate,h0R)
        o_tm1=words[0]
       
        errSent=e
        for i in xrange(len(words)-1):
            word=words[i+1]
            expectedVector=expectedVectors[i+1]
            [wordMaxP,h_tm1LM,o_tm1]=srnnLM.classify(item.input,h_tm1LM,o_tm1)
            #===================================================================
            # for index in xrange(len(o_tm1)):
            #     if o_tm1[index]<0.12:
            #         o_tm1[index]=0
            #===================================================================
            [e,h_tm1RR,_]=srnnRR.train(item.input,o_tm1,expectedVector,learningRate,h_tm1RR)
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
    
def evaluateSRNNUID(srnnLM,srnnRR,outFile, evalSet, h0,o0,h0R,periods):
    predictions_test=getModelPredictionsUID(srnnLM,srnnRR,evalSet,h0,o0,h0R,periods)
    
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

def UIDLoss(probsSeq):
    mean=0
    for prob in probsSeq:
        mean+=prob
    mean=mean*1.0/len(probsSeq)
    
    sd=0
    for prob in probsSeq:
        sd+=(prob-mean)*(prob-mean)
        
    sd=sd*1.0/len(probsSeq)

    return numpy.sqrt(sd)

    
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
             'lr':float(sys.argv[postIncre("x")]), #learning rate
             'verbose':int(sys.argv[postIncre("x")]),
             'decay':int(sys.argv[postIncre("x")]), # decay on the learning rate if improvement stops
             'nhiddenLM':int(sys.argv[postIncre("x")]), # number of hidden units for language layer
             'nhiddenRR':int(sys.argv[postIncre("x")]), # number of hiiden units for reranker layer
             'seed':int(sys.argv[postIncre("x")]),
             
             'percTrain':float(sys.argv[postIncre("x")]),
             'percValidate':float(sys.argv[postIncre("x")]),
             'percTest':float(sys.argv[postIncre("x")]),
             
             'nepochs':int(sys.argv[postIncre("x")]),
             'label':sys.argv[postIncre("x")],#"dotsPaper_dssSeparated_811_1",
             'periods':int(sys.argv[postIncre("x")]),
             #'loadLM':int(sys.argv[13]), #Whether the model is already trained or not
             'loadRR':int(sys.argv[postIncre("x")]),#whether the reranker model was already trained or not
             'inputType':sys.argv[postIncre("x")], #dss or sitVector or compVector
             'actpas':sys.argv[postIncre("x")],
             'inputFile':sys.argv[postIncre("x")], #FILE containing the corpus input
             'lmFolder':sys.argv[postIncre("x")] #folder containing the already trained production network      
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
         #'label':"dotsPaper_dssSeparated_trainstop_normalInitialization_monitor1_4",#"dotsPaper_dssSeparated_811_1",
         'label':"15_40_trainstop_monitor_25K_UID_sigm_paper",#unnormalized2-1-1540
         'periods':True,
         'loadRR':True,
         'inputType':'beliefVector', #dss or sitVector or compVector
         'actpas':True,
         'inputFile':corpusListsPath, #FILE containing the corpus input
         'lmProductionFolder':modelProductionPath#"/Users/jesus/Documents/EclipseWorkspace/productionModel/outputs/prod_main_mon_5cond_outputs/output_beliefVector_120h_0.124lr_200ep_dots_15_40_trainstop_monitor_25K_sch/lastModel"
         }

    if s['periods']: s['vocab_size']=43
    else: s['vocab_size']=42
    
    if s['inputType']=='sitVector' or s['inputType']=='compVector' or s['inputType']=="beliefVector": s['inputDimension']=44
    if s['inputType']=='dss': s['inputDimension']=150
    if s['actpas']:s['inputDimension']=s['inputDimension']+1
    
    #LOAD FILES
    mapIndexWord=loadFiles.getWordLocalistMap(wordLocalistMapPath)
    matrix,events=loadFiles.getAtomicEventDSSMap(dsssMatrixPath)
    
    corpusUID=CorpusUID()
    corpusUID.loadFromPickle(s['inputFile'])
    
    trainList=corpusUID.training
    testList=corpusUID.testing
    trainTest=corpusUID.trainTest

    loadFiles.addPeriods(trainList)
    for item in trainList:
        loadFiles.addPeriods(item.equivalents)
    loadFiles.setInputType(trainList,s['inputType']) 
    
    
    loadFiles.addPeriods(testList)
    for item in testList:
        loadFiles.addPeriods(item.equivalents)
    loadFiles.setInputType(testList,s['inputType'])
    
    
    loadFiles.addPeriods(trainTest)
    for item in trainTest:
        loadFiles.addPeriods(item.equivalents)
    
    loadFiles.setInputType(trainTest,s['inputType']) 
    
    validate=trainTest

    folderThisRun,bestModelUID,lastModelUID=getFolders(outputsPath,s)
    random.seed(s['seed'])
    
    #CREATE LANGUAGE LEARNING SRNN  
    srnn = prodSRNN_notBPTT_mon.model(
                              inputDimens=s['inputDimension'],
                              hiddenDimens = s['nhiddenLM'],
                              outputDimens= s['vocab_size']
                     )  
    h0=numpy.zeros(s['nhiddenLM'], dtype=theano.config.floatX)
    o0=numpy.zeros(s['vocab_size'], dtype=theano.config.floatX)
    srnn.load(s['lmProductionFolder'])
    #We assume that the network already knows how to produce language, now we want it to learn how to rerarnk according to UID

    #CREATE RERANKER SRNN
    srnnRR = prodSRNN_notBPTT_UID.model(
                              inputDimens=s['vocab_size'],
                              hiddenDimens = s['nhiddenRR'],
                              outputDimens= s['vocab_size'],
                              dssDimens=    s['inputDimension']
                 )
    h0R=numpy.zeros(s['nhiddenRR'], dtype=theano.config.floatX)
    
    #IF THE RERANKER HASNT BEEN TRAINED 
    if not s['loadRR']:   
        outputFile= open(folderThisRun+'/outputRR.txt', 'w+')
        best_sim = -numpy.inf
        bestEp=0 
        epErrors=[]
        epSimilarities=[]
        s['clr'] = s['lr']
        
        for epoch in xrange(s['nepochs']):
            random.shuffle(trainList)
           
            #TRAIN THIS EPOCH  
            errors=epochTrainRR(srnn,srnnRR,trainList,s['clr'],h0,o0,h0R) 
            epErrors.append(sum(errors))
            
            #HERE IS THE DIFFERENCE BETWEEN TRAINING NUMBERS OF THE TRAIN SET AND TESTING NUMBERS ALSO OF THE TRAIN SET (the periods flag)
            predictions_validate=getModelPredictionsUID(srnn,srnnRR,validate,h0,o0,h0R,0) #We dont stop on periods, because at the beginning the system may not know that it has to put a period
              
            #Get a list of pairs (sim,mostSimilar) where sim is the similarity of the most similar sentence (mostSimilar) in the gold sentences of the given dss 
            simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(validate,predictions_validate)]
            #Get only the list of similarities
            similarities=[sim for (sim,mostSimilar) in simgolds]
            similarity=numpy.sum(similarities)/len(similarities)    
            epSimilarities.append(similarity)
               
            if similarity > best_sim:
                srnnRR.save(bestModelUID)
                best_sim = similarity
                bestEp=epoch
                lastGood=epoch#just an aux variable that we can change while keeping track of bestEp
                          
                outputFile.write('NEW BEST EPOCH: '+str(epoch)+' similarity: '+str(similarity)+"\n")
                print 'NEW BEST: epoch', epoch, 'similarity', similarity
                
            else:
                outputFile.write('Epoch: '+str(epoch)+' lr: '+str(s['clr'])+' similarity: '+str(similarity)+'\n')
                print 'Epoch:',epoch,'lr:',s['clr'],'similarity:',similarity      
                 
            errorsPlot=plt.figure(5000)
            plt.plot(epErrors)
            plt.savefig(folderThisRun+"/errorsTrainEpUID.png")
                          
            simPlot=plt.figure(1000000)
            plt.plot(epSimilarities)
            plt.savefig(folderThisRun+"/similaritiesUID.png")
                 
            # learning rate halves if no improvement in 10 epochs
            if s['decay'] and (epoch-lastGood) >= 10: 
                s['clr'] *= 0.5
                lastGood=epoch#Here we have to reset lastGood, otherwise it will halve each epoch until we get an improvement
                
            #TRAINING STOPS IF THE LEARNING RATE IS BELOW THRESHOLD OR IF NO IMPROVEMENT DURING 30 EPOCHS
            if s['clr'] < 2e-3 or (epoch-bestEp)>=20:     
                break   
             
        srnnRR.save(lastModelUID)
        
        print 'BEST RESULT: epoch', bestEp, 'Similarity:', best_sim,'with the model', folderThisRun
        outputFile.write('BEST RESULT: epoch '+str(bestEp)+' Similarity: '+str(best_sim)+' with the model '+folderThisRun)
        outputFile.close()
        
    else:   #IF THE MODEL WAS ALREADY TRAINED AND WE ARE ONLY LOADING IT FOR TESTING
        import matplotlib.pyplot as plt
        import numpy as np

        
        srnnRR.load(lastModelUID)        
        outFileTrainUID= open(folderThisRun+'/outputlast_trainUID.txt', 'w+')

        evaluateSRNNUID(srnn,srnnRR,outFileTrainUID, trainTest, h0, o0,h0R,s['periods'])
        outFileTrainUID.close()  
        slopeDataP=[]
        slopeDataStd=[]
        slopeDataAcc=[]
        #slopes=numpy.arange(20,350,20)
        slopes=numpy.arange(1,5,0.5)
        interceps=numpy.arange(1,5,0.5)
        #slopes=[130]
        #interceps=[2.5]
        for paramSlope in slopes:
            print "SLOPE:"+str(paramSlope)
            InterDataP=[]
            InterDataStd=[]
            InterDataAcc=[]
            for paramInterc in interceps:
                
                modelPredictions=getModelPredictionsUIDTest(srnn,srnnRR,testList,h0,o0,h0R,1,paramInterc,paramSlope)
                simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(testList,modelPredictions)]
                similarities=[acc for (acc,gold) in simgolds]
                golds=[gold for (acc,gold) in simgolds]
                
                
                   
                predWords=[indicesToSentence(pred,mapIndexWord) for pred in modelPredictions]
                labelWords=[indicesToSentence(label,mapIndexWord) for label in golds]
                accuracyGlobal=numpy.sum(similarities)/len(similarities)
                derlens=[len(pred) for pred in modelPredictions]
                print numpy.mean(derlens)
                
                #Create Comprehender Language Model
                srnnComp = langmodelSRNN_notBPTT.model(
                                      inputDimens=s['vocab_size'],
                                      hiddenDimens = s['nhiddenLM'],
                                      outputDimens= s['vocab_size']
                             )  
                srnnComp.load(modelSurprisalPath)
                w0=numpy.zeros(s['vocab_size'], dtype=theano.config.floatX)
                 
                #SURPRISAL VALUES
                probabilities=[]
                surprisalValues=[]
                wordsI=[]
                for sent in modelPredictions:
                    probLangModel=getSentenceProbs(srnnComp,sent,h0,w0)   
                    probabilities.append(probLangModel)
                
                #print surprisals
                for sentence in probabilities:
                        for (word,wIndex,surpValue) in sentence:
                            surprisalValues.append(numpy.log(surpValue)*-1)
                            wordsI.append(word)
                #=======================================================================
                # for word,surprisal in zip(wordsI,surprisalValues):
                #     print word,surprisal
                #=======================================================================
                
                #===============================================================
                # print "\tInter:"+str(paramInterc)
                # print "\tP:"+str(numpy.mean(surprisalValues))
                # print "\tStd:"+str(numpy.std(surprisalValues))
                # print
                #===============================================================
                InterDataP.append(numpy.mean(surprisalValues))
                InterDataStd.append(numpy.std(surprisalValues))
                InterDataAcc.append(accuracyGlobal)
            slopeDataP.append(InterDataP)
            slopeDataStd.append(InterDataStd)
            slopeDataAcc.append(InterDataAcc)
        print slopeDataP
        print slopeDataStd
        print slopeDataAcc
        plt.imshow(slopeDataP, cmap='hot', interpolation='none')
        plt.show()
        plt.imshow(slopeDataStd,cmap='hot', interpolation='none')
        plt.show()
        plt.imshow(slopeDataAcc,cmap='hot', interpolation='none')
        plt.show()
        
        
        
        #=======================================================================
        # maxDSS=0
        # for dss in testList:
        #     if len(dss.equivalents)>maxDSS:
        #         maxDSS=len(dss.equivalents)
        #         maxDSSvalue= dss
        # print maxDSS
        # print maxDSSvalue.testItem
        #=======================================================================

   #============================================================================
   #      decod=SentenceDecoderUID(srnn,srnnRR,h0,o0,h0R)
   # 
   #      for item in testList:
   #          print
   #          sentences=decod.getNBestPredictedSentencesPerDSS(item,0.1)
   #          #tree=DerivationTree(item)
   #          print len(sentences)
   #             
   #          #SENTENCES AND RANKING ACCORDING TO PRODUCTION MODEL
   #          sumaModel=0.0
   #          for sent in sentences:
   #              sentWords=indicesToSentence(sent.indices,mapIndexWord)
   #              print sentWords
   #              print str(sent.probability)
   #              sumaModel+=sent.probability   
   #                    
   #          print 
   #          print "sumProdModelPs:"+str(sumaModel)    
   #      
   #============================================================================

     #==========================================================================
     #            #PREFIX PROBABILITIES ACCORDING TO SRNN LM
     #            probsSRNN=[]
     #            for sent in item.equivalents:
     #                probLangModel= getSentenceProb(lmsrnn,sent,h0)
     #                probsSRNN.append((sent.testItem,probLangModel,sent.schema))
     #            #sortedListSRNN=sorted(probsSRNN,key=lambda tup:tup[1],reverse=True)
     #            
     #            #SCHEMA PROBABILITIES GIVEN THE DSS ACCORDING TO FFNN
     #            schemaLoc=numpy.zeros(51, dtype=theano.config.floatX)
     #            schemaLoc[int(item.schema)-1]=1.0            
     #            [pred,loss]=ffnn.classify(item.input,schemaLoc)
     #            schCondProbs={}
     #            for x in xrange(51):
     #                schCondProbs[x+1]=pred[x]
     #             
     #                            
     #            tree.getTreeProbs(probs)
     #             
     #            #PREFIX PROBABILITIES ACCORDING TO TREE AND SCHEMAProbs
     #            sumaTreeProbs=0.0
     #            sentsTreeP=[]
     #            for equiv in item.equivalents:
     #                sent=equiv.testItem
     #                prob= tree.decodeSentProb(sent)
     #                sentsTreeP.append((sent,prob))
     #                sumaTreeProbs+=prob
     #            print "sumTreePs:"+str(sumaTreeProbs)
     #            #sortedList=sorted(sentsTreeP,key=lambda tup:tup[1],reverse=True)
     #                 
     #                 
     #            #ATTEMPT AT COMBINING DIFFERENT INFORMATION
     #            print 
     #            sortedList3=[]
     #            for (sent,probLM,schema),(sent1,probTree) in zip(probsSRNN,sentsTreeP):
     #                schemaP=pred[int(schema)-1]
     #                probfinal=probTree*probLM #     prob2=50.0*prob1+prob
     #                sortedList3.append((sent,probfinal))
     #                 
     #            sortedList3=sorted(sortedList3,key=lambda tup:tup[1],reverse=True)
     #            for (sent,prob) in sortedList3:
     #                print sent
     #                print prob
     # 
     #==========================================================================
     
     
     



