'''
Created on Apr 1, 2016

@author: jesus

Classes and methods to manipulate Trees. Each tree is obtained based on the sentences that
are related to a DSS. The root node is the start of sentence, then each following node is 
a possible word continuation, leaves are periods.
'''
import numpy,theano
from collections import Counter


class SimpleDerivationTree:
    '''
    Takes a list of sentences in string form where each token is separated by a space
    and forms a derivation tree. 
    '''
    def __init__(self,sentsList):
        agenda=[]
        for sent in sentsList:
            sentence=sent.split()
            agenda.append(sentence)
        
        rootNode=SimpleTreeNode("ROOT",[],agenda)
        rootNode.processIncompleteNode()
        self.root=rootNode
        
    def printMe(self):
        self.root.printMe()
    

class SimpleTreeNode:
    '''
    Each node of a SimpleDerivationTree
    Contains one word, links to children and if the tree is under construction,
    an agenda that contains the remaining children to be processed
    '''
    def __init__(self,firstWord,children,agenda):
        self.word=firstWord
        self.children=children
        self.agenda=agenda
        
    def printMe(self,printChildren=True):
        print self.word
        print self.agenda
        
        if printChildren:
            print "CHILDREN:"
            for child in self.children:
                child.printMe()
            
    def processIncompleteNode(self):
        while len(self.agenda)>0:
            sentence=self.agenda[0]
            if len(sentence)<1:
                self.agenda.pop(0)
                continue
            word=sentence[0]
            sufix=sentence[1:]
            newAgendaItem=sufix
            
            currNewNode=SimpleTreeNode(word,[],[newAgendaItem])
            self.children.append(currNewNode)
            self.agenda.pop(0)
            for sufix in self.agenda[:]:
                if sufix[0]==currNewNode.word:
                    newSufix=sufix[1:]
                    newAgendaItem=newSufix
                    currNewNode.agenda.append(newAgendaItem)
                    self.agenda.remove(sufix)
        for child in self.children:
            child.processIncompleteNode()
        return True
    
    
  
class DerivationTree:
    '''
    Takes a list of TrainingElement and creates a derivation tree based on the sentences
    Similar to SimpleDerivationTree, but with much richer information
    '''  
    def __init__(self,trainingElementList):
        agenda=[]
        schemas=[]
        for sent in trainingElementList:
            sentence=sent.testItem.split()
            agenda.append((sentence,sent.schema))
            schemas.append(sent.schema)
        
        self.root=TreeNode("ROOT",[],schemas,agenda)
        self.root.condP=1.0
        self.root.processIncompleteNode()
        
    
    def printMe(self):
        self.root.printMe()
        
    def printMeFull(self):
        self.root.printMeFull()
      
      
    def processDerLengthsInfo(self,mapWordToIndex):
        self.root.getNodeDepth(-1)  #setup node depths
        self.root.getNodeMinDerLength()#get minimum derivation lengths
        self.root.getNodeMaxMinDerLength()#get maximum of the minderlenghts of the children
        self.root.derLenWeigth=1.0#initialize root's derlenW
        self.root.getChildrenDerLenWeigths()#compute derlenWs for children
        self.root.getDerLenWeigthsVector(mapWordToIndex)#get vectors
      
    
    def decodeSentProb(self,sentence):
        words=sentence.split()
        currentNode=self.root
        prefixP=1.0
        for word in words:
            for child in currentNode.children:
                if child.word==word:
                    prefixP*=child.condP
                    currentNode=child
                    break
        return prefixP
    
    def decodeSentDerLenProb(self,sentence):
        words=sentence.split()
        currentNode=self.root
        prefixP=1.0
        for word in words:
            for child in currentNode.children:
                if child.word==word:
                    prefixP*=child.derLenWeigth
                    currentNode=child
                    break
        return prefixP 
    
    def getLengthTrainingVectors(self,sentence):
        words=sentence.split()
        currentNode=self.root
        trainingVectors=[]
        trainingVectors.append(self.root.derLenWVector)
        for word in words:
            for child in currentNode.children:
                if child.word==word:
                    if word is not ".":
                        trainingVectors.append(child.derLenWVector)
                    currentNode=child
                    break
        return trainingVectors
    

class TreeNode:
    '''
    Each node of a DerivationTree
    '''
    def __init__(self,firstWord,children,schemas,agenda):
        self.word=firstWord
        self.agenda=agenda
        self.schemas=schemas
        self.children=children
        
    def printMe(self,printChildren=True):
        print
        print self.word
        print "Agenda:",self.agenda
        print "Schemas:",self.schemas
        print "Dept:",self.depth
        print "CondP",self.condP
        if printChildren and len(self.children)>0:
            print "CHILDREN:"
            for child in self.children:
                child.printMe()
            
    def printMeFull(self,printChildren=True):
        print
        self.printMe(False)
        print "MinDerLen:",self.minDerLen
        print "MaxMinDerLen:",self.maxMinDerLen
        print "weight:"+str(self.derLenWeigth)
        if hasattr(self, 'derLenWVector'):
            print "vector:"
            print self.derLenWVector
              
        if printChildren and len(self.children)>0:
            print "CHILDREN:"
            for child in self.children:
                child.printMeFull()
        
            
    def processIncompleteNode(self):
        self.nSents=len(self.agenda)
        while len(self.agenda)>0:
            (sentence,schema)=self.agenda[0]
            if len(sentence)<1:
                self.agenda.pop(0)
                continue
            
            word=sentence[0]
            sufix=sentence[1:]
            newAgItem=(sufix,schema)
            currNewNode=TreeNode(word,[],[schema],[newAgItem])
            self.children.append(currNewNode)
            self.agenda.pop(0)
            
            for (sufix,schema) in self.agenda[:]:
                if sufix[0]==currNewNode.word:
                    newSufix=sufix[1:]
                    newAgItem=(newSufix,schema)
                    currNewNode.agenda.append(newAgItem)
                    currNewNode.schemas.append(schema)
                    self.agenda.remove((sufix,schema))
        for child in self.children:
            child.condP=len(child.agenda)*1.0/self.nSents
            child.processIncompleteNode()
        return True

            
    def getNodeDepth(self,motherDepth):
        self.depth=motherDepth+1
        for child in self.children:
            child.getNodeDepth(self.depth)
    
    def getNodeMinDerLength(self):
        if len(self.children)==0:
            self.minDerLen=self.depth
        else:
            self.minDerLen=self.children[0].getNodeMinDerLength()
            for x in xrange(1,len(self.children)):
                otherDerLen=self.children[x].getNodeMinDerLength()
                if otherDerLen<self.minDerLen:
                    self.minDerLen=otherDerLen
        return self.minDerLen
    def getNodeMaxMinDerLength(self):
        if len(self.children)>0:
            maxMin=self.children[0].minDerLen
            for x in xrange(1,len(self.children)):
                otherMinDerLen=self.children[x].minDerLen
                if otherMinDerLen>maxMin:
                    maxMin=otherMinDerLen
            self.maxMinDerLen=maxMin
            for child in self.children:
                child.getNodeMaxMinDerLength()
        else:
            self.maxMinDerLen=self.minDerLen

    def getChildrenDerLenWeigths(self):
        if len(self.children)==0:
            return
        if len(self.children)==1: #If there is only one child, then all probability goes to it
            self.children[0].derLenWeigth=1.0
        else:
            z=0 #normalization constant
            for child in self.children:
                child.derLenWeigth=self.maxMinDerLen-child.minDerLen+1
                z+=child.derLenWeigth
            z=z*1.0 #to make it float
            for child in self.children: #normalization
                child.derLenWeigth=child.derLenWeigth/z
                
        for child in self.children:#recursion to children
            child.getChildrenDerLenWeigths()
    
    def getDerLenWeigthsVector(self,mapWordToIndex):
        if len(self.children)==0:return #leaves are always periods ".", which means there should be no new vectors
        
        vocabSize=len(mapWordToIndex) 
        self.derLenWVector=numpy.zeros(vocabSize, dtype=theano.config.floatX)  # @UndefinedVariable
        for child in self.children:
            self.derLenWVector[mapWordToIndex[child.word]]=child.derLenWeigth
            child.getDerLenWeigthsVector(mapWordToIndex)
            
        

class TreeComparer:
    '''
    Compares two SimpleDerivationTree instances 
    Needs further development
    '''
    def __init__(self):
        self.countOver=Counter()
        self.countUnder=Counter()
        
    def flush(self):
        self.countOver=Counter()
        self.countUnder=Counter()
    
    def nodeCompare(self,node1,node2):
        for child2 in node2.children:
            found=False
            for child1 in node1.children:
                if child2.word==child1.word:
                    found=True
                    break
            if not found:
                if child2.word=="a":
                    if len(child2.children)>0:
                        for chichild in child2.children:
                            bigram="a "+chichild.word
                            if self.countOver.has_key(bigram):
                                self.countOver[bigram]+=1
                            else:self.countOver[bigram]=1
                if child2.word=="in":
                    if len(child2.children)>0:
                        for chichild in child2.children:
                            bigram="in "+chichild.word
                            if bigram=="in the":
                                if len(chichild.children)>0:
                                    for chichichild in chichild.children:
                                        trigram="in the "+chichichild.word
                                        if self.countOver.has_key(trigram):
                                            self.countOver[trigram]+=1
                                        else:self.countOver[trigram]=1
                            
                if self.countOver.has_key(child2.word):
                    self.countOver[child2.word]+=1
                else:self.countOver[child2.word]=1
                #print "overgeneratingx: "+child2.word
            
        for child1 in node1.children:
            found=False
            for child2 in node2.children:
                if child1.word==child2.word:
                    self.nodeCompare(child1, child2)
                    found=True
                    break
            if not found:
                if child1.word=="a":
                    if len(child1.children)>0:
                        for chichild in child1.children:
                            bigram="a "+chichild.word
                            if self.countUnder.has_key(bigram):
                                self.countUnder[bigram]+=1
                            else:self.countUnder[bigram]=1
                if child1.word=="in":
                    if len(child1.children)>0:
                        for chichild in child1.children:
                            bigram="in "+chichild.word
                            if bigram=="in the":
                                if len(chichild.children)>0:
                                    for chichichild in chichild.children:
                                        trigram="in the "+chichichild.word
                                        if self.countUnder.has_key(trigram):
                                            self.countUnder[trigram]+=1
                                        else:self.countUnder[trigram]=1
                if self.countUnder.has_key(child1.word):
                    self.countUnder[child1.word]+=1
                else:self.countUnder[child1.word]=1
                #print "undergeneratingx: "+child1.word
            
                
        
        
        


if __name__ == '__main__':
    
#===============================================================================
#     #TEST CREATING TWO SIMPLEDERIVATIONTREES AND COMPARING THEM
#     sentList=[['heidi', 'plays', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'inside', '.'],['a', 'girl', 'plays', 'inside', 'with', 'a', 'toy', '.'],['a', 'girl', 'plays', 'with', 'a', 'toy', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'inside', 'with', 'a', 'jigsaw', '.'],['a', 'girl', 'plays', 'inside', 'with', 'a', 'puzzle', '.'],['a', 'girl', 'plays', 'with', 'a', 'jigsaw', '.'],['a', 'girl', 'plays', 'with', 'a', 'puzzle', '.'],['a', 'girl', 'plays', 'with', 'a', 'toy', '.'],['a', 'girl', 'plays', 'with', 'a', 'puzzle', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'with', 'a', 'jigsaw', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'with', 'a', 'toy', 'inside', '.']]
#     sentsList=[" ".join(sent) for sent in sentList]
#     print sentsList
# 
#     sentList2=[['sophia', 'plays', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'inside', '.'],['a', 'girl', 'plays', 'inside', 'with', 'a', 'toy', '.'],['a', 'girl', 'plays', 'with', 'a', 'toy', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'inside', 'with', 'a', 'jigsaw', '.'],['a', 'girl', 'plays', 'inside', 'with', 'a', 'puzzle', '.'],['a', 'girl', 'plays', 'with', 'a', 'jigsaw', '.'],['a', 'girl', 'plays', 'with', 'a', 'puzzle', '.'],['a', 'girl', 'plays', 'with', 'a', 'toy', '.'],['a', 'girl', 'plays', 'with', 'a', 'puzzle', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'with', 'a', 'jigsaw', 'in', 'the', 'bedroom', '.'],['a', 'girl', 'plays', 'with', 'a', 'toy', 'inside', '.']]
#     sentsList2=[" ".join(sent) for sent in sentList2]
#     print sentsList2
#     
#     
#     tree=SimpleDerivationTree(sentsList)
#     tree2=SimpleDerivationTree(sentsList2)
#     
#     comparer=TreeComparer()
#     comparer.nodeCompare(tree.root, tree2.root)
#===============================================================================
    
#===============================================================================
#     corpusFilePath="../data/dataFiles/filesSchemas_with150DSS_withSims96/corpusUID.pick"
#     corpusListsPath2="../data/dataFiles/filesSchemas_with150DSS_withSims96/corpus_UID_imbalancedProbs.pick"
#     
#     testSentence="someone plays ."
# 
#     from data.containers import CorpusLM
#     corpusLM=CorpusLM()
#     
#     corpusLM.loadFromPickle(corpusFilePath)
#     testTree1=DerivationTree(corpusLM.training)
# 
#     corpusLM.loadFromPickle(corpusListsPath2)
#     testTree2=DerivationTree(corpusLM.training)
#     
#     print testTree1.decodeSentProb2(testSentence)
#     print testTree2.decodeSentProb2(testSentence)
#     
#     exit()
#===============================================================================
    
    wordLocalistMapPath='dataFiles/map_localist_words.txt'
    from data.loadFiles import getWordLocalistMap,getMapWordToIndex
    from data.crossValidation import Fold
    
    inputFile="dataFiles/filesSchemas/trainTest_Conditions_finalSchemas_0.pick"
    fold=Fold()
    fold.loadFromPickle(inputFile)
    
    mapIndexWord=getWordLocalistMap(wordLocalistMapPath)
    mapWordToIndex=getMapWordToIndex(mapIndexWord)
     
    trainSets=fold.trainSet # 2 lists, one for training, one for validating
    testSets=fold.valtestSet #7 conditions, each with 14 dss
    
    oneItem=trainSets[1][2] 
    for equiv in oneItem.equivalents:
        equiv.testItem=equiv.testItem+" ."
        print equiv.testItem
        
  
    tree=DerivationTree(oneItem.equivalents)
    tree.processDerLengthsInfo(mapWordToIndex)

    tuples=[]
    for sent in oneItem.equivalents:
        print sent.testItem
        print sent.lengthVectors
        sentP=tree.decodeSentProb(sent.testItem)
        
        sentLP=tree.decodeSentDerLenProb(sent.testItem)
        vectors=tree.getLengthTrainingVectors(sent.testItem)
        print sentP
        print sentLP
        #print vectors
        print sent.wordsLocalist
        tu=(sent.testItem,sentLP)
        tuples.append(tu)
        
        sortedList=sorted(tuples,key=lambda tup:tup[1],reverse=True)
    for (sent,prob) in sortedList:
        print sent
        print prob 

    