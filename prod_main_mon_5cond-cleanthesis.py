import theano, numpy
import random, os, sys

import matplotlib.pyplot as plt
import data.loadFiles as loadFiles
from data.crossValidation import Fold
from tools.similarities import levenSimilarity
from production.decoder import SentenceDecoder
 
import rnn.prodSRNN_notBPTT_mon as prodSRNN_notBPTT_mon

sys.path.append("../data")
corpusListsPath="../data/filesSchemas_with150DSS_withSims96/trainTest_Conditions_finalSchemasWithSimilars96_0.pick"

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

def wordsToIndices(words,mapping):
    return[mapping[word] for word in words]
    
def getEquivSentencesIndicesSet(trainElem,mapping):
    sentenceWordsSet=[equiv.testItem for equiv in trainElem.equivalents]
    sentenceSplitWordsSet=[sentence.split() for sentence in sentenceWordsSet]  
    return [wordsToIndices(wordsSplitted,mapping) for wordsSplitted in sentenceSplitWordsSet]

def getModelPredictions(srnn,testSet,h0,o0,periods):
    predictions_test=[] 
    
    for sentence in testSet:
        predSent=[]
        
        [predWord,h_tm1,o_tm1]=srnn.classify(sentence.input,h0,o0) 
        #predWord is the index of the highest activated word in the output, h_tm1 is the content of the 
        #hidden layer and o_tm1 is the output layer, ie, prob distribution
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


"""
    Creates the 3 folders where all results/models will be stored
    folderThisRun: folder containing all the files of this particular run, will be contained in 
                   folderThisModel which contains all runs of this specific Python file
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


if __name__ == '__main__':

    if len(sys.argv)>1:
        x=1
        s={
             'lr':float(sys.argv[xplusplus("x")]),#learning rate
             'verbose':int(sys.argv[xplusplus("x")]),
             'decay':int(sys.argv[xplusplus("x")]), # decay on the learning rate if improvement stops
             'nhidden':int(sys.argv[xplusplus("x")]), # number of hidden units
             'seed':int(sys.argv[xplusplus("x")]), #random number generator seed
             
             'percTrain':float(sys.argv[xplusplus("x")]),
             'percValidate':float(sys.argv[xplusplus("x")]),
             'percTest':float(sys.argv[xplusplus("x")]),
             
             'nepochs':int(sys.argv[xplusplus("x")]), #maximum number of epochs
             'label':sys.argv[xplusplus("x")],#label for current run
             'periods':int(sys.argv[xplusplus("x")]), #whether or not the sentences contain periods
             'load':int(sys.argv[xplusplus("x")]), #Whether the model is already trained or not
             'inputType':sys.argv[xplusplus("x")], #dss or sitVector or compVector
             'actpas':sys.argv[xplusplus("x")], #whether we differentiate active from passive sentences
             'inputFile':sys.argv[xplusplus("x")] #FILE containing the corpus input
         }
           
    else:
        s = {
         'lr':0.24, #learning rate
         'verbose':1, 
         'decay':True, #decay on the learning rate if improvement stops
         'nhidden':120, #number of hidden units
         'seed':345, #random number generator seed
         
         'percTrain':0.9,
         'percValidate':0.0,
         'percTest':0.1,
         
         'nepochs':200, #maximum number of epochs
         'label':"15_40_monitor_sigm_0", #label for current run
         'periods':True, #whether or not the sentences contain periods
         'load':True, #whether the model is already trained or not
         'inputType':'beliefVector', #dss or sitVector or compVector
         'actpas':True, #whether we differentiate active from passive sentences
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
    
    for listT in testList:
        loadFiles.addPeriods(listT)
        for item in listT:
            loadFiles.addPeriods(item.equivalents)
        loadFiles.setInputType(listT,s['inputType'])
    
    train=trainLists[0]
    trainTest=trainLists[1]
    validate=trainTest
    
    folderThisRun,bestModel,lastModel,plotsFolder=getFolders(outputsPath,s)
     
    #CREATE SRNN AND INITIALIZE
    srnn = prodSRNN_notBPTT_mon.model(
                              inputDimens=s['inputDimension'],
                              hiddenDimens = s['nhidden'],
                              outputDimens= s['vocab_size']
                     )        
    
    h0=numpy.zeros(s['nhidden'], dtype=theano.config.floatX)
    o0=numpy.zeros(s['vocab_size'], dtype=theano.config.floatX)
    random.seed(s['seed'])
    
    #IF THE MODEL HASN'T BEEN TRAINED 
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
            
            predictions_validate=getModelPredictions(srnn,validate,h0,o0,0) 
            #We dont stop on periods, because at the beginning the system may not know that it has to put a period
             
            #Get a list of pairs (sim,mostSimilar) where sim is the similarity of the most similar sentence (mostSimilar) 
            #in the gold sentences of the given dss 
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
      
    else: 
        #IF THE MODEL WAS ALREADY TRAINED AND WE ARE ONLY LOADING IT FOR TESTING
        srnn.load(lastModel)

        outFileTrain= open(folderThisRun+'/outputlast_train.txt', 'w+')
        outFileTest= open(folderThisRun+'/outputlast_test.txt', 'w+')

        evaluateSRNN(srnn, outFileTrain, trainTest, h0, o0, s['periods'])
        outFileTrain.close()
         
        for index in xrange(len(testList)):
            print "\nCONDITION:"+str(index+1)+"\n"
            outFileTest.write("\nCONDITION:"+str(index+1)+"\n")
            evaluateSRNN(srnn, outFileTest, testList[index],  h0, o0, s['periods'])
            
        outFileTest.close()
         
        

