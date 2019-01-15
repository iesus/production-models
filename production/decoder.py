'''
Created on Apr 20, 2016

@author: jesus
'''
import math, os
import matplotlib.pyplot as plt
import numpy as np

class ScoreIndexMap:
    def __init__(self, elements=[]):
        self.elements=elements
    def printMe(self):
        for elem in self.elements:
            elem.printMe()
            
    def quickSortElements(self,lo,hi):
        def partition(lo,hi):
            pivot=self.elements[lo].score
            i=lo-1
            j=hi+1
            while i<j:
                i+=1
                while self.elements[i].score>pivot:
                    i+=1
                j-=1
                while self.elements[j].score<pivot:
                    j=j-1    
                if i>=j:
                    return j
                aux=self.elements[i]
                self.elements[i]=self.elements[j]
                self.elements[j]=aux
        if lo< hi:
            p=partition(lo,hi)
            self.quickSortElements(lo,p)
            self.quickSortElements(p+1,hi)
        return self.elements
    
    def getOrderedWordsList(self,indexWordMap):
        sentenceWordIndices=[elem.index for elem in self.elements]
        words=indicesToWords(sentenceWordIndices,indexWordMap)
        return words
    
    def plotWordProbs(self,indexWordMap,label,path):
        words=self.getOrderedWordsList(indexWordMap)   
        #getOutputPlot(mapWords,words,zeroLabel+"_0",plotsDSS+"/"+zeroLabel+"_0.png")
        scoreValues=[elem.score for elem in self.elements]
        #indicesWords=[elem.index for elem in mapWords.elements]
        
        indicesPlot=np.arange(len(self.elements))
        bar_width = 0.35    
        fig = plt.figure(666)
        ax = fig.add_subplot(111)
        
        ax.bar(indicesPlot, scoreValues, bar_width,color='b')
        
        ax.set_ylabel("Probability")
        ax.set_title(label,fontsize=8)
        ax.set_xticks(indicesPlot+bar_width/2)
        ax.set_xticklabels(words,rotation=90)
        
        fig.set_tight_layout(True)
        
        fig.savefig(path)
        fig.clear()

    
    def getFromOutputLayer(self,outLayer):
        ind=0
        scoreIndexList=[]
        for score in outLayer:
            scoreIndex=ScoreIndex(score,ind)
            ind+=1
            scoreIndexList.append(scoreIndex)
        self.elements=scoreIndexList
        #mapInd=ScoreIndexMap(scoreIndexList)
        #return self
        
    def getEntropy(self):
        ent=0.0
        for scoreIndex in self.elements:
            logP=math.log(scoreIndex.score)
            ent+=scoreIndex.score*logP
        return ent
    
    def getExpProbability(self):
        expP=0.0
        for scoreIndex in self.elements:
            expP+=scoreIndex.score*scoreIndex.score
        return expP
        
    
    def getBestScores(self, param):
        bestScores=[]
        #entP=math.exp(self.getEntropy()) #entropy based scores
        #expP=self.getExpProbability() #scores with expected probabilities
        
        #maxP=self.elements[0].score 
        thresh=param#*maxP
        for scoreIndex in self.elements:
            if scoreIndex.score>thresh:
                bestScores.append(scoreIndex)
            else: break
        return bestScores
    
    
    
class ScoreIndex:
    def __init__(self,score,index):
        self.score=score
        self.index=index
    def printMe(self):
        print("Index:")+str(self.index)
        print("Score:")+str(self.score)


class ProducedPrefix:
    def __init__(self,prob,indicesList,h_tm1,treeNode):
        self.probability=prob
        self.indices=indicesList
        self.lastWord=indicesList[-1]
        self.h_tm1=h_tm1
        self.treeNode=treeNode
        
    def copy(self):
        prob=self.probability
        indicesList=self.indices[:]
        h_tm1=self.h_tm1.copy() 
        newCopy=ProducedPrefix(prob,indicesList,h_tm1)
        return newCopy
    def printMe(self):
        print self.probability
        print self.indices
        print self.h_tm1
        print self.lastWord
        
class ProducedPrefixUID:
    def __init__(self,prob,indicesList,h_tm1,h_tm1RR):
        self.probability=prob
        self.indices=indicesList
        self.lastWord=indicesList[-1]
        self.h_tm1=h_tm1
        self.h_tm1RR=h_tm1RR
    def copy(self):
        prob=self.probability
        indicesList=self.indices[:]
        h_tm1=self.h_tm1.copy()
        h_tm1RR=self.h_tm1RR.copy()
        newCopy=ProducedPrefixUID(prob,indicesList,h_tm1,h_tm1RR)
        return newCopy
    def printMe(self):
        print self.probability
        print self.indices
        print self.h_tm1
        print self.h_tm1RR
        print self.lastWord
 
def indicesToWords(indices,indexWordMapping):
    return [indexWordMapping[index] for index in indices]
 
def getOutputPlot(mapWords,words,label,path):
    scoreValues=[elem.score for elem in mapWords.elements]
    #indicesWords=[elem.index for elem in mapWords.elements]
    
    indicesPlot=np.arange(len(mapWords.elements))
    bar_width = 0.35    
    fig = plt.figure(666)
    ax = fig.add_subplot(111)
    
    ax.bar(indicesPlot, scoreValues, bar_width,color='b')
    
    ax.set_ylabel("Probability")
    ax.set_title(label,fontsize=8)
    ax.set_xticks(indicesPlot+bar_width/2)
    ax.set_xticklabels(words,rotation=90)
    
    fig.set_tight_layout(True)
    
    fig.savefig(path)
    fig.clear()


class SentenceDecoder:
    
    def __init__(self,srnn,mapIndexWord):
        self.srnn=srnn
        self.h0=srnn.h0
        self.o0=srnn.o0
        self.indexWordMap=mapIndexWord
        
    def getNBestPredictedSentencesPerDSS(self,testItem,param,plotsFolder="none"):            
        productionAgenda=[]
        productions=[]
        
        if plotsFolder=="none":boolPlots=False
        else:boolPlots=True
        
        if boolPlots:
            plotsDSS=plotsFolder+"/"+testItem.testItem.replace(" ","_")
            if not os.path.exists(plotsDSS): os.mkdir(plotsDSS)
            zeroLabel=testItem.testItem[:-2]
            
        [_,h_tm1,o_tm1]=self.srnn.classify(testItem.input,self.h0,self.o0)
        mapWords=ScoreIndexMap()
        mapWords.getFromOutputLayer(o_tm1)
        mapWords.quickSortElements(0, len(mapWords.elements)-1)#If we dont want to plot with sorted words, we can just plot before the quicksort
        
        if boolPlots:mapWords.plotWordProbs(self.indexWordMap,zeroLabel+" 0",plotsDSS+"/"+zeroLabel+"_0.png")
       
        bestScores=mapWords.getBestScores(param)
        for scoreIndex in bestScores:            
            oneProd=ProducedPrefix(scoreIndex.score,[scoreIndex.index],h_tm1,None)
            productionAgenda.append(oneProd)

        while len(productionAgenda)>0:
            prefix=productionAgenda.pop(0)
            if prefix.lastWord>41 or len(prefix.indices)>15: 
                productions.append(prefix)
                continue
            
            o_tm1=self.o0.copy()
            o_tm1[prefix.lastWord]=1.0
            
            [_,h_tm1,o_tm1]=self.srnn.classify(testItem.input,prefix.h_tm1,o_tm1)
            mapWords.getFromOutputLayer(o_tm1)
            mapWords.quickSortElements(0, len(mapWords.elements)-1)
            
            if boolPlots:
                prefixWords=indicesToWords(prefix.indices,self.indexWordMap)
                prefixWordsTogether="_".join(prefixWords)
                mapWords.plotWordProbs(self.indexWordMap,zeroLabel+"\n"+prefixWordsTogether,plotsDSS+"/"+zeroLabel+"_"+prefixWordsTogether+".png")
            
            bestScores=mapWords.getBestScores(param)
            for scoreIndex in bestScores:    
                newIndicesChain=prefix.indices[:]
                newProb=prefix.probability*scoreIndex.score
                newIndicesChain.append(scoreIndex.index)
                
                newPrefix=ProducedPrefix(newProb,newIndicesChain,h_tm1,None)
                productionAgenda.append(newPrefix)
            
        productions.sort(key=lambda x: x.probability, reverse=True)
        return productions
    
    
    #DEPRECATED, IT'S BETTER TO USE getNBestPredictedSentencesPerDSS AND TreeComparer INSTEAD
    def getNBestPredictedSentencesPerDSSLookingForMistakes(self,testItem,param,plotsFolder,boolPlots,treeRoot):     
        def lookForNode(word,motherNode):
            if motherNode is None: return None
            for child in motherNode.children:
                if child.word==word:
                    return child
            
        productionAgenda=[]
        productions=[]
        
        from collections import Counter
        overgenerations=Counter({})
        undergenerations=Counter({})
        
        if boolPlots:
            plotsDSS=plotsFolder+"/"+testItem.testItem.replace(" ","_")
            if not os.path.exists(plotsDSS): os.mkdir(plotsDSS)
            zeroLabel=testItem.testItem[:-2]
            
        [_,h_tm1,o_tm1]=self.srnn.classify(testItem.input,self.h0,self.o0)
        mapWords=ScoreIndexMap()
        mapWords.getFromOutputLayer(o_tm1)
        mapWords.quickSortElements(0, len(mapWords.elements)-1)#If we dont want to plot with sorted words, we can just plot before the quicksort
        
        if boolPlots:mapWords.plotWordProbs(self.indexWordMap,zeroLabel+" 0",plotsDSS+"/"+zeroLabel+"_0.png")
       
        bestScores=mapWords.getBestScores(param)  
        suffixes=[child.word for child in treeRoot.children]
        
        for scoreIndex in bestScores:
            word = self.indexWordMap[scoreIndex.index]
            childNode=lookForNode(word,treeRoot)
            if word in suffixes: suffixes.remove(word)
            else: 
                if overgenerations.has_key(word): overgenerations[word]+=1
                else: overgenerations[word]=1
                
            oneProd=ProducedPrefix(scoreIndex.score,[scoreIndex.index],h_tm1,childNode) 
            productionAgenda.append(oneProd)
        
        for suffix in suffixes:
            if undergenerations.has_key(suffix): undergenerations[suffix]+=1
            else: undergenerations[suffix]=1
            
        while len(productionAgenda)>0:
            prefix=productionAgenda.pop(0)
            treeNode=prefix.treeNode
            
            if prefix.lastWord>41 or len(prefix.indices)>15: 
                productions.append(prefix)
                continue
            
            o_tm1=self.o0.copy()
            o_tm1[prefix.lastWord]=1.0
            
            [_,h_tm1,o_tm1]=self.srnn.classify(testItem.input,prefix.h_tm1,o_tm1)
            mapWords.getFromOutputLayer(o_tm1)
            mapWords.quickSortElements(0, len(mapWords.elements)-1)
            
            if boolPlots:
                prefixWords=indicesToWords(prefix.indices,self.indexWordMap)
                prefixWordsTogether="_".join(prefixWords)
                mapWords.plotWordProbs(self.indexWordMap,zeroLabel+"\n"+prefixWordsTogether,plotsDSS+"/"+zeroLabel+"_"+prefixWordsTogether+".png")
            
            bestScores=mapWords.getBestScores(param)
            suffixes=[]
            if treeNode !=None:
                for child in treeNode.children:
                    suffixes.append(child.word)
                    
            for scoreIndex in bestScores:    
                newIndicesChain=prefix.indices[:]
                newProb=prefix.probability*scoreIndex.score
                newIndicesChain.append(scoreIndex.index)
                
                word=self.indexWordMap[scoreIndex.index]
                childNode=lookForNode(word,treeNode)
                if word in suffixes: suffixes.remove(word)
                elif treeNode != None: 
                    if overgenerations.has_key(word): overgenerations[word]+=1
                    else: overgenerations[word]=1
                    
                newPrefix=ProducedPrefix(newProb,newIndicesChain,h_tm1,childNode)
                productionAgenda.append(newPrefix)
            
            for suffix in suffixes:
                if undergenerations.has_key(suffix): undergenerations[suffix]+=1
                else: undergenerations[suffix]=1
            
        productions.sort(key=lambda x: x.probability, reverse=True)
        return productions,overgenerations,undergenerations
         
         
class SentenceDecoderUID:
    def __init__(self,srnnRR):
        self.srnnLM=srnnRR.simpleProdModel
        self.srnnRR=srnnRR
        self.h0=srnnRR.simpleProdModel.h0
        self.o0=srnnRR.o0
        self.h0R=srnnRR.h0R
        
    def getNBestPredictedSentencesPerDSS(self,testItem,param):
        productionAgenda=[]
        productions=[]
        
        [_,h_tm1,o_tm1]=self.srnnLM.classify(testItem.input,self.h0,self.o0)
        [h_tm1R,predOutDistro]=self.srnnRR.classify(testItem.input,o_tm1,self.h0R)
        
        mapWords=ScoreIndexMap()
        mapWords.getFromOutputLayer(predOutDistro)
        mapWords.quickSortElements(0, len(mapWords.elements)-1)
        
        bestScores=mapWords.getBestScores(param)
        for scoreIndex in bestScores:
            oneProd=ProducedPrefixUID(scoreIndex.score,[scoreIndex.index],h_tm1,h_tm1R)
            productionAgenda.append(oneProd)
    
        while len(productionAgenda)>0:
            prefix=productionAgenda.pop(0)
            if prefix.lastWord>41 or len(prefix.indices)>15: 
                productions.append(prefix)
                continue
            
            o_tm1=self.o0.copy()
            o_tm1[prefix.lastWord]=1.0
            
            [_,h_tm1,o_tm1]=self.srnnLM.classify(testItem.input,prefix.h_tm1,o_tm1)
            [h_tm1R,predOutDistro]=self.srnnRR.classify(testItem.input,o_tm1,prefix.h_tm1RR)
            mapWords.getFromOutputLayer(predOutDistro)
            mapWords.quickSortElements(0, len(mapWords.elements)-1)
            bestScores=mapWords.getBestScores(param)
            
            for scoreIndex in bestScores:    
                newIndicesChain=prefix.indices[:]
                newProb=prefix.probability*scoreIndex.score
                newIndicesChain.append(scoreIndex.index)
                
                newPrefix=ProducedPrefixUID(newProb,newIndicesChain,h_tm1,h_tm1R)
                productionAgenda.append(newPrefix)
            
        productions.sort(key=lambda x: x.probability, reverse=True)
        return productions
    
    def getUIDLoss(self,testItem,sentIndices):
        pass
        
        
        
        