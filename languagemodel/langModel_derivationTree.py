'''
Created on Apr 1, 2016

@author: jesus
'''
import sys
sys.path.append("../data")

'''
Takes a set of sentences and creates a language model based on the derivation tree that can be obtained from the sentences
'''

class DerivationTree:
    def __init__(self,itemsList):
        agenda=[]
        for sent in itemsList:
            sentence=sent.testItem.split()
            agenda.append(sentence)
        
        self.root=TreeNode("ROOT",[],agenda)
        self.root.condP=1.0
        self.root.processIncompleteNode()
          
    def printMe(self):
        self.root.printMe()

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
    

class TreeNode:
    def __init__(self,firstWord,children,agenda):
        self.word=firstWord
        self.agenda=agenda
        self.children=children
        
    def printMe(self,printChildren=True):
        print self.word
        print self.nSents
        print self.condP
        if printChildren:
            print "CHILDREN:"
            for child in self.children:
                child.printMe()
            
    def processIncompleteNode(self):
        self.nSents=len(self.agenda)
       
        while len(self.agenda)>0:
            sentence=self.agenda[0]
            if len(sentence)<1:
                self.agenda.pop(0)
                continue
            
            word=sentence[0]
            sufix=sentence[1:]
            
            currNewNode=TreeNode(word,[],[sufix])
            self.children.append(currNewNode)   
            self.agenda.pop(0)
            
            for sufix in self.agenda[:]:
                if sufix[0]==currNewNode.word:
                    newSufix=sufix[1:]
                    currNewNode.agenda.append(newSufix)
                    self.agenda.remove(sufix)
                    
        for child in self.children:  
            child.condP=len(child.agenda)*1.0/self.nSents
            child.processIncompleteNode()
        
        
    

                



if __name__ == '__main__':
    
    corpusFilePath="../data/dataFiles/filesSchemas_with150DSS_withSims96/corpusUID.pick"
    corpusListsPath2="../data/dataFiles/filesSchemas_with150DSS_withSims96/corpus_UID_imbalancedProbs.pick"

    #testSentence="heidi plays soccer in the street ."
    testSentence="someone plays ."

    from data.containers import CorpusLM
    corpusLM=CorpusLM()
    
    corpusLM.loadFromPickle(corpusFilePath)
    testTree1=DerivationTree(corpusLM.training)

    

    testTree1.printMe()    
    print testTree1.decodeSentProb(testSentence)
    
    corpusLM.loadFromPickle(corpusListsPath2)
    testTree2=DerivationTree(corpusLM.training)
    print testTree2.decodeSentProb(testSentence)




    