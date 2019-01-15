'''
Created on Apr 8, 2016

@author: jesus calvillo
Contains methods related to clustering of DSSs in the corpus
'''

from operator import itemgetter
from tools.similarities import cosineSimilarity


'''
Each cluster obtained by measuring similarity of value
elements is the list of members of that cluster
'''
class Cluster:
    def __init__(self,value):
        self.value=value
        self.elements=[]


'''
Get the k most semantically similar items in a corpus 
corpus is a list of items with a ".dss" attribute, which is used to calculate similarity
a ".simSents" attribute is added to said items 
'''
def getSemanticSimilarsDSS(corpus,k):
    '''
    Takes an item with a ".simSents" attribute (a list of similar items with size k)
    adds an item to simSents and reorders it accordingly to preserve the maximum k elements
    '''
    def addSimilarItem(currentItem, newSimilar,similarity, beamSize):
        if len(currentItem.simSents)<beamSize:
            currentItem.simSents.append((newSimilar,similarity))
            currentItem.simSents=sorted(currentItem.simSents, key=itemgetter(1),reverse=True)
        else:
            if similarity>currentItem.simSents[beamSize-1][1]:
                currentItem.simSents[beamSize-1]=(newSimilar,similarity)
                currentItem.simSents=sorted(currentItem.simSents, key=itemgetter(1),reverse=True)    
    
    
    for i in xrange(len(corpus)):
        corpus[i].simSents=[]
        for y in xrange(len(corpus)):
            if y==i:continue
            sim=cosineSimilarity(corpus[i].dss,corpus[y].dss)
            addSimilarItem(corpus[i],corpus[y],sim,k)            


'''
Takes a list of elements (elementList), where each element has to have 
an attribute ".value". Based on that attribute, clustering is performed using cosine similarity
where a cluster is formed with elements with similarity at least simValue
'''
def getClustersCosine(elementList,simValue):
    clusterList=[]
    
    for element in elementList:
        match=False
        for cluster in clusterList:
            if cosineSimilarity(element.value,cluster.value)>simValue: 
                match=cluster
                cluster.elements.append(element)
                break
            
        if not match: 
            newCluster=Cluster(element.value)
            newCluster.elements.append(element)
            clusterList.append(newCluster)
    return clusterList

'''
Receives a list of Situation clusters and prints it
'''
def printSituationClusters(situationClustersList):
    for situationCluster in situationClustersList:
        
        if len(situationCluster.elements)<2:continue #if it's a singleton continue
        
        for situation in situationCluster.elements:
            print situation.actives[0].testItem
            for item in situation.actives[1:]:
                print "\t"+item.testItem
            if len(situation.passives)>0:
                for item in situation.passives:
                    print "\t"+item.testItem 
        print
    print len(situationClustersList)




if __name__ == '__main__':  
    
    from containers import CorpusAP
    corpusAPFinal=CorpusAP()
    corpusAPFinal.loadFromPickle("dataFiles/filesSchemas_with150DSS/corpusAPFinal_schemas.pick")
         
    sitAPs=corpusAPFinal.act+ corpusAPFinal.actpas    
     
    clusters=getClustersCosine(sitAPs,0.94)
    printSituationClusters(clusters)
    print len(clusters)

