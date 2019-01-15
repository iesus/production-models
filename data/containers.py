'''
Created on Apr 8, 2016

@author: jesus calvillo
Contains definitions for the data structures that are used throughout these packages.
'''
import cPickle

'''
A situation refers to a semantic representation. It is usually linked to several sentences (TrainingElements)
''' 
class Situation:
    def __init__(self, value, actives=[],passives=[]):
        self.value=value            #semantic vector related to the situation
        self.actives=actives        #list of active TrainingElements
        self.passives=passives      #list of passive TrainingElements

'''
Single unit used for training. Each TrainingElement is related to one sentence in the corpus.
Contains the semantic representation, a list of other TrainingElements with equivalent semantics, among other information
'''
class TrainingElement:   
    def __init__(self,schema,testItem,numberOfWords,semantics,wordsLocalist,dss,active=True):
        self.testItem=testItem    #sentence
        self.schema=schema        #grammar rule used to generate the sentence
        self.numberOfWords=numberOfWords 
        self.semantics=semantics  #semantic representation in logical form
        self.wordsLocalist=wordsLocalist #sentence translated to a sequence of localist vectors
        self.active=active        #whether sentence is in active or pasive voice
        self.DSSValue=dss         #DSS vector
        
    def printMe(self):
        print self.testItem
        print self.semantics
        print self.DSSValue
        
'''
Container used after loading the raw corpus. Each instance contains the sentences related to a DSS, either only passives or actives
'''
class InputObjectAP:
    def __init__(self,value,ap):
        self.active=ap#active: ap=1 passive: ap=0
        self.value=value
        self.sentences=[]
   
class InputObject:
    def __init__(self,value):
        self.value=value
        self.sentences=[]     
'''
Corpus that collapses all dss into just one list of TrainingElement instances
'''
class Corpus:
    def __init__(self,elements=[]):
        self.elements=elements
    
    def saveToFileReadable(self, filePath):
        with open(filePath,'w') as outputFile:
            for item in self.elements:
                outputFile.write(item.testItem+"\n")
                outputFile.write(item.semantics+"\n")
                outputFile.write(str(item.dss)+"\n")
                outputFile.write(str(item.sitVector)+"\n")
                outputFile.write("\n")
    def saveToPickle(self, filePath):
        outFile=file(filePath,'wb')
        cPickle.dump(self.elements,outFile,protocol=cPickle.HIGHEST_PROTOCOL)
        outFile.close()
    def loadFromPickle(self,filePath):
        inputFile=file(filePath,'rb')
        self.elements=cPickle.load(inputFile)
        inputFile.close()
        
'''
Corpus that separates situations with actives and passives, from situations with only actives.
'''        
class CorpusAP:
    def __init__(self,actpas=[],act=[]):
        self.actpas=actpas  #SITUATIONS THAT HAVE ACTIVE AND PASSIVE REALIZATIONS
        self.act=act        #SITUATIONS THAT ONLY HAVE ACTIVE REALIZATIONS
        
    def saveToPickle(self, filePath):
        outFile=file(filePath,'wb')
        cPickle.dump(self.actpas,outFile,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.act,outFile,protocol=cPickle.HIGHEST_PROTOCOL)
        outFile.close()
        
    def loadFromPickle(self,filePath):
        inputFile=file(filePath,'rb')
        self.actpas=cPickle.load(inputFile)
        self.act=cPickle.load(inputFile)
        inputFile.close()
        
    def getSentencesFile(self,outputFilePath):
        outFile=open(outputFilePath,'w+')
        for sit in self.actpas:
            for te in sit.actives:
                outFile.write(te.testItem+"\n")
            for te in sit.passives:
                outFile.write(te.testItem+"\n")
        for sit in self.act:
            for te in sit.actives:
                outFile.write(te.testItem+"\n")        
        outFile.close()
        
    def getReadableCorpusFile(self,outputFilePath):
        with open(outputFilePath,'w') as outputFile:
            outputFile.write("SITUATIONS WITH ACTIVE AND PASSIVE SENTENCES \n\n")
            for item in self.actpas:
                outputFile.write("\n"+str(item.value)+"\n")
                outputFile.write("Actives:\n")
                for realact in item.actives:
                    outputFile.write("\t"+realact.testItem+"\n")
                outputFile.write("Passives:\n")
                for realpas in item.passives:
                    outputFile.write("\t"+realpas.testItem+"\n")   
            
            outputFile.write("SITUATIONS WITH ONLY ACTIVE SENTENCES \n\n")
            for item in self.act:
                outputFile.write("\n"+str(item.value)+"\n")
                outputFile.write("Actives:\n")
                for realact in item.actives:
                    outputFile.write("\t"+realact.testItem+"\n")
        
'''
Corpus used to train/test a Language Model
Contains 3 lists: training, validation and testing

Also used to train the UID model, in that case 
there are 8201 sentences and 968 DSS representations (splitting actives from passives)
'''                    
class CorpusLM:
    def __init__(self,training=[],validation=[],testing=[]):
        self.training=training  #Contains all training sentences in the corpus
        self.validation=validation#uid:Contains one sentence per dss in the training set, used to test production accuracy in the uid model
        self.testing=testing    #uid:Contains only the DSS representations related to more than 1 sentence
        
        
    def saveToPickle(self, filePath):
        outFile=file(filePath,'wb')
        cPickle.dump(self.training,outFile,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.validation,outFile,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.testing,outFile,protocol=cPickle.HIGHEST_PROTOCOL)
        outFile.close()
        
    def loadFromPickle(self,filePath):
        inputFile=file(filePath,'rb')
        self.training=cPickle.load(inputFile)
        self.validation=cPickle.load(inputFile)
        self.testing=cPickle.load(inputFile)
        inputFile.close()