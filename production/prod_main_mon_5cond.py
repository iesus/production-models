import numpy,random, os, sys
import matplotlib.pyplot as plt

import data.loadFiles as loadFiles
from data.crossValidation import Fold
from tools.similarities import levenSimilarity
from tools.plusplus import xplusplus 
import rnn.prodSRNN_notBPTT_mon 

sys.path.append("../data")
corpusFilePath="../data/dataFiles/files-thesis/trainTest_Cond-thesis_0.pick"
wordLocalistMapPath='../data/dataFiles/map_localist_words.txt'
outputsPath="../outputs"


def localistToIndices(localistMatrix):
    return [numpy.argmax(localist) for localist in localistMatrix]

def indicesToWords(indices,indexWordMapping):
    return [indexWordMapping[index] for index in indices]

def wordsToIndices(words,wordIndexMapping):
    return[wordIndexMapping[word] for word in words]

def getEquivSentencesIndicesSet(trainElem):
    return [localistToIndices(equivalent.wordsLocalist) for equivalent in trainElem.equivalents]



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



def mostSimilarEquivalentsLevens(trainingElement,modelProduction):
    '''
    Compares the sentence produced by the model with the set of possible sentences related to the DSS,
    obtains the most similar one with its similarity score
    '''
    #Get the possible sentences using word indices
    equivalentsIndices=[localistToIndices(equivalent.wordsLocalist) for equivalent in trainingElement.equivalents]
    #Compare each possible sentence with the sentence the model produced
    similarities=[levenSimilarity(eq,modelProduction) for eq in equivalentsIndices]
    #Get the most similar one
    mostSimilar=numpy.argmax(similarities, 0)
    
    return (similarities[mostSimilar],equivalentsIndices[mostSimilar])


def evaluateSRNN(srnn, outFile, evalSet):
    productions_test=srnn.getModelProductions(evalSet)
    
    simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(evalSet,productions_test)]
    similarities=[acc for (acc,_) in simgolds]
    golds=[gold for (_,gold) in simgolds]
       
    predWords=[indicesToWords(pred,mapIndexWord) for pred in productions_test]
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



def testAllFolds_Precision_Recall_Fscore():
    '''
    Having trained 10 models on 10 different folds, test the model in terms of precision, recall and fscore
    when trying to produce all sentences for each DSS
    '''
    import data.setAnalysis as setAnalyzer  
    from decoder import SentenceDecoder
    
    foldsValues=[[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]
    goodsSetSizes=[]
    badsSetSizes=[]
     
    fold=Fold() 
    prefixModelsPath="../outputs/prod_main_mon_5cond_outputs/systematicity_journal_paper/output_beliefVector_120h_0.24lr_200ep_dots_5cond_25K_"
    prefixFoldsPath="../data/dataFiles/serverData2/filesSchemas_with150DSS_withSims96/trainTest_Conditions_finalSchemasWithSimilars96_"
    
    
    for x in xrange(10):
        modelPath= prefixModelsPath+str(x)+"/lastModel"
        srnn.load(modelPath)
        
        foldPath=prefixFoldsPath+str(x)+".pick" 
        fold.loadFromPickle(foldPath) #not sure why we need InputObject defined in containers, because of old fold object
        
        testLists=fold.valtestSet
        for tlist in testLists:
            for elem in tlist:
                loadFiles.addPeriods(elem.equivalents,42)#needed because the version of the corpus is old, a new version would not need this line
            loadFiles.setInputType(tlist,s['inputType'])
        
        decod=SentenceDecoder(srnn,mapIndexWord) 
        listasCond=[testLists[0],testLists[1],testLists[2],testLists[3],testLists[5]]
         
        for index in xrange(len(listasCond)): 
            lista=listasCond[index]
            allValues=[[],[],[]]
             
            bads=0  #goods are 70-bad
            for item in lista:
                indicesSet=getEquivSentencesIndicesSet(item) 
                sentencesModel=decod.getNBestPredictedSentencesPerDSS(item,0.12)
                sentencesModelIndices=[sent.indices for sent in sentencesModel]
            
                prec,rec,fscore=setAnalyzer.precisionRecallFScore(indicesSet,sentencesModelIndices)
                
                if fscore<1.0:
                    bads+=1
                    badsSetSizes.append(len(item.equivalents))
                else:
                    goodsSetSizes.append(len(item.equivalents))
                
                allValues[0].append(prec)
                allValues[1].append(rec)
                allValues[2].append(fscore)
                
            avPrec=numpy.mean(allValues[0], axis=0)
            avRec=numpy.mean(allValues[1], axis=0)
            avFsc=numpy.mean(allValues[2], axis=0)
             
            foldsValues[0][index].append(avPrec)
            foldsValues[1][index].append(avRec)
            foldsValues[2][index].append(avFsc)
            foldsValues[3][index].append(14-bads)
        
    condPrec= numpy.mean(foldsValues[0],axis=1)
    condRec= numpy.mean(foldsValues[1],axis=1)
    condFSc= numpy.mean(foldsValues[2],axis=1)
    condPerf=numpy.mean(foldsValues[3],axis=1)/14.0
     
    avPrecA=numpy.mean(condPrec)
    avRecA=numpy.mean(condRec)
    avFscA=numpy.mean(condFSc)
    avPerf=numpy.mean(condPerf)
     
    avGoodSetSize=numpy.mean(goodsSetSizes)
    sdGoodSetSize=numpy.std(goodsSetSizes)
    avBadSetSize=numpy.mean(badsSetSizes)
    sdBadSetSize=numpy.std(badsSetSizes)
     
    #=======================================================================
    # avPrecA=numpy.mean(numpy.take(condPrec,[0,1,2,3,5]))
    # avRecA=numpy.mean(numpy.take(condRec,[0,1,2,3,5]))
    # avFscA=numpy.mean(numpy.take(condFSc,[0,1,2,3,5]))
    #=======================================================================
     
    print condPrec
    print avPrecA
    
    print condRec
    print avRecA
     
    print condFSc
    print avFscA
     
    print condPerf
    print avPerf
     
    print goodsSetSizes
    print avGoodSetSize
    print sdGoodSetSize
    print
    print badsSetSizes
    print avBadSetSize
    print sdBadSetSize

if __name__ == '__main__':

    if len(sys.argv)>1:
        x=1
        s={
             'lr':float(sys.argv[xplusplus("x")]),      #learning rate
             'decay':int(sys.argv[xplusplus("x")]),     #decay on the learning rate if improvement stops
             'nhidden':int(sys.argv[xplusplus("x")]),   #number of hidden units
             'seed':int(sys.argv[xplusplus("x")]),      #seed for random 
             'nepochs':int(sys.argv[xplusplus("x")]),   #max number of training epochs
             'label':sys.argv[xplusplus("x")],          #label for this run
             'periods':int(sys.argv[xplusplus("x")]),   #whether the corpus has periods
             'load':int(sys.argv[xplusplus("x")]),      #whether the model is already trained or not
             'inputType':sys.argv[xplusplus("x")],      #dss or sitVector or compVector
             'actpas':sys.argv[xplusplus("x")],         #if the inputs are divided in actpas
             'inputFile':sys.argv[xplusplus("x")]       #FILE containing the input data
         }
           
    else:
        s = {
         'lr':0.24,                 #learning rate 
         'decay':True,              #decay on the learning rate if improvement stops
         'nhidden':120,             #number of hidden units
         'seed':345,                #seed for random
         'nepochs':200,             #max number of training epochs
         'label':"15_40_monitor_sigm_anew1",     #label for this run
         'periods':True,            #whether the corpus has periods
         'load':True,               #whether the model is already trained or not
         'inputType':'beliefVector',#dss or sitVector or compVector
         'actpas':True,             #if the inputs are divided in actpas
         'inputFile':corpusFilePath   #FILE containing the input data
         }
    if s['periods']: s['vocab_size']=43
    else: s['vocab_size']=42
    
    if s['inputType']=='sitVector' or s['inputType']=='compVector' or s['inputType']=="beliefVector": s['inputDimension']=44
    if s['inputType']=='dss': s['inputDimension']=150
    if s['actpas']:s['inputDimension']=s['inputDimension']+1
    
    #LOAD FILES
    mapIndexWord=loadFiles.getWordLocalistMap(wordLocalistMapPath)
    
    fold=Fold()
    fold.loadFromPickle(s['inputFile'])
    trainLists=fold.trainSet
    testLists=fold.valtestSet

    loadFiles.setInputType(trainLists[0],s['inputType'])
    for tList in testLists:
        loadFiles.setInputType(tList,s['inputType'])
    
    train=trainLists[0]
    validateList=trainLists[1]# Traintest is used instead of validation
    
    folderThisRun,bestModel,lastModel,plotsFolder=getFolders(outputsPath,s)
     
    #CREATE SRNN AND INITIALIZE VARS
    srnn = rnn.prodSRNN_notBPTT_mon.model(
                              inputDimens=s['inputDimension'],
                              hiddenDimens = s['nhidden'],
                              outputDimens= s['vocab_size']
                     )        
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
            errors=srnn.epochTrain(train,s['clr'])
            epErrors.append(sum(errors))
            
            predictions_validate=srnn.getModelProductions(validateList,False)#We don't stop on periods, because at the beginning the model may not know that it has to put a period
             
            #Get a list of pairs (sim,mostSimilar) where sim is the similarity of the most similar sentence (mostSimilar) in the gold sentences of the given dss 
            simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(validateList,predictions_validate)]
            #Get only the list of similarities
            similarities=[sim for (sim,mostSimilar) in simgolds]
            similarity=numpy.sum(similarities)/len(similarities)    
            epSimilarities.append(similarity)    
            
            outputLine='Epoch: '+str(epoch)+' lr: '+str(s['clr'])+' similarity: '+str(similarity)
            
            if similarity > best_sim:
                srnn.save(bestModel)
                best_sim = similarity
                bestEp=epoch
                lastChange_LR=epoch#just an aux variable that we can change while keeping track of bestEp         
                outputLine='NEW BEST '+outputLine
            
            outputFile.write(outputLine+'\n')
            print outputLine
                   
            errorsPlot=plt.figure(100000)
            plt.plot(epErrors)
            plt.savefig(folderThisRun+"/errorsTrainEp.png")
                          
            simPlot=plt.figure(1000000)
            plt.plot(epSimilarities)
            plt.savefig(folderThisRun+"/similarities.png")
                
            # learning rate halves if no improvement in 15 epochs
            if s['decay'] and (epoch-lastChange_LR) >= 15: 
                s['clr'] *= 0.5
                lastChange_LR=epoch#we have to reset lastChange_LR, otherwise it will halve each epoch until we get an improvement
                
            #TRAINING STOPS IF THE LEARNING RATE IS BELOW THRESHOLD OR IF NO IMPROVEMENT DURING 40 EPOCHS
            if s['clr'] < 1e-3 or (epoch-bestEp)>=40:     
                break  
            
        srnn.save(lastModel)
        outputLine='BEST RESULT: epoch '+str(bestEp)+' Similarity: '+str(best_sim)+' with the model '+folderThisRun
        print outputLine
        outputFile.write(outputLine)
        outputFile.close()
      
    else: #IF THE MODEL WAS ALREADY TRAINED AND WE ARE ONLY LOADING IT FOR TESTING

        #=======================================================================
        # srnn.load(lastModel)  
        # outFileTrain= open(folderThisRun+'/outputlast_train.txt', 'w+')
        # outFileTest= open(folderThisRun+'/outputlast_test.txt', 'w+')
        #   
        # evaluateSRNN(srnn, outFileTrain, validateList)
        # outFileTrain.close()
        #  
        # for index in xrange(len(testLists)):
        #     print "\nCONDITION:"+str(index+1)+"\n"
        #     outFileTest.write("\nCONDITION:"+str(index+1)+"\n")
        #     evaluateSRNN(srnn, outFileTest, testLists[index])     
        # outFileTest.close()
        #=======================================================================
       


        #=======================================================================
        # TESTING WITH DEEPER ANALYSIS
        # CODE BELOW ASSUMES K MODELS HAVE BEEN TRAINED CORRESPONDING TO K FOLDS OF A CROSSVALIDATION SCHEME
        # NEEDS CLEANING AND FURTHER DEVELOPMENT
        #=======================================================================
        
        import data.setAnalysis as setAnalyzer
        from decoder import SentenceDecoder
        from data.derivationTreeDSS import SimpleDerivationTree, TreeComparer
        from collections import Counter
        
        mapWordIndex={word:index for index,word in mapIndexWord.items()}
        
        overgenerationsAll=Counter({})
        undergenerationsAll=Counter({})
        countgood=0
        countbad=0
        
        verybads=[]   
        sitsAGame=[]
          
        outFile= open('/Users/jzc1104/Desktop/outputPassives.txt', 'w+')
        
        fold=Fold() 
        prefixModelsPath="../outputs/prod_main_mon_5cond_outputs/systematicity_journal_paper/output_beliefVector_120h_0.24lr_200ep_dots_5cond_25K_"
        prefixFoldsPath="../data/dataFiles/serverData2/filesSchemas_with150DSS_withSims96/trainTest_Conditions_finalSchemasWithSimilars96_"   
           
        for x in xrange(3):       
            outFile.write("NEW FOLD\n")
            
            modelPath= prefixModelsPath+str(x)+"/lastModel"
            srnn.load(modelPath)
            decod=SentenceDecoder(srnn,mapIndexWord)
        
            foldPath=prefixFoldsPath+str(x)+".pick" 
            fold.loadFromPickle(foldPath) #not sure why we need InputObject defined in containers, because of old fold object
        
            testLists=fold.valtestSet        
            for lista in testLists:
                for item in lista:
                    loadFiles.addPeriods(item.equivalents,42)
                loadFiles.setInputType(lista,s['inputType'])
                
            overgenerationsFold=Counter({})
            undergenerationsFold=Counter({})
            comparer=TreeComparer()
              
            testconds=[testLists[0],testLists[1],testLists[2],testLists[3],testLists[5]]
            testPassives=[testLists[4],testLists[6]]
           
            for lista in testconds:
                outFile.write("NEW LIST:\n")
                                
                for item in lista:
                        
                    #Get expected sentences and the corresponding derivation tree
                    sentsExpected=[equiv.testItem for equiv in item.equivalents]
                    tree1=SimpleDerivationTree(sentsExpected)
                      
                    #Get model's predictions and the corresponding derivation tree
                    sentences=decod.getNBestPredictedSentencesPerDSS(item,0.12)  
                    sentencesModelIndices=[sent.indices for sent in sentences] 
                    sentencesWordsSplit=[indicesToWords(sent.indices,mapIndexWord) for sent in sentences]
                    sentencesStringList=[" ".join(splitSent) for splitSent in sentencesWordsSplit]
                    tree2=SimpleDerivationTree(sentencesStringList)
                                         
                    #Compare the 2 trees and count over and undergenerations
                    comparer.nodeCompare(tree1.root, tree2.root)
                    overgenerationsFold=comparer.countOver+overgenerationsFold
                    undergenerationsFold=comparer.countUnder+undergenerationsFold
                    
                    indicesSet=getEquivSentencesIndicesSet(item)                    
                    
                    def printDifferences(expectedIndices,producedIndices, producedProbabilities, mapIndexWord,outFile):
                        print "Expected:"
                        outFile.write("Expected:\n")
                        
                        for expected in expectedIndices:
                            sentence=" ".join([mapIndexWord[index] for index in expected])
                            print sentence
                            outFile.write(sentence+"\n")
                        outFile.write("\n")
                        print len(expectedIndices)
                        
                        sumaModel=0.0
                        print "Produced:"
                        for sent,prob in zip(producedIndices,producedProbabilities):
                            sentWords=" ".join([mapIndexWord[index] for index in sent])
                            print str(prob)+"    \t"+str(sentWords)
                            sumaModel+=prob
                        print len(producedIndices)
                        print "sumProdModelPs:"+str(sumaModel)  
                        
                        print
                        for generated in producedIndices:
                            if generated not in expectedIndices:
                                sentWords=" ".join([mapIndexWord[index] for index in generated])
                                outFile.write("\tover:\t"+sentWords+"\n")
                                print "overgenerated: "+sentWords
                        print
                        for expec in expectedIndices:
                            if expec not in producedIndices:
                                sentWords=" ".join([mapIndexWord[index] for index in expec])
                                outFile.write("\t\tunder:\t"+sentWords+"\n")
                                print "undergenerated: "+sentWords
                        
                              
                    prec,rec,fscore=setAnalyzer.precisionRecallFScore(indicesSet,sentencesModelIndices)
                    if fscore<1.0:#if the performance was less than perfect
                            print
                            print comparer.countOver
                            print comparer.countUnder
                            
                            print "PROBLEMATIC!!!"
                            countbad+=1
                            
                            probabilities=[sent.probability for sent in sentences]
                            printDifferences(indicesSet, sentencesModelIndices, probabilities, mapIndexWord, outFile)
      
                    else: countgood+=1
                    if rec==0.0:
                        verybads.append((item.equivalents,sentencesStringList))
                          
                    if comparer.countUnder.has_key("at"):sitsAGame.append((item.equivalents,sentencesStringList))       
                    comparer.flush()
           
            print
            print "Fold counters:"
            print overgenerationsFold
            print undergenerationsFold    
            overgenerationsAll=overgenerationsAll+overgenerationsFold
            undergenerationsAll=undergenerationsAll+undergenerationsFold
               
        print
        print "Global counters:"
        print overgenerationsAll
        print undergenerationsAll
        print countgood
        print countbad
       
        outFile.close()
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
