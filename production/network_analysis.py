'''
Created on Oct 20, 2017
@author: jesus


This assumes a network defined by: prodSRNN_notBPTT_mon
We analyze the weights and activations, based on Layer-Wise Relevance Propagation
'''
import matplotlib.pyplot as plt
import rnn.prodSRNN_notBPTT_mon as prodSRNN_notBPTT_mon
import numpy, heapq, plotly

class WordInfoAnalysis:
    def __init__(self,index,word,activationVector,weightsVector,relAct,relInh,relevance):
        self.index=index
        self.word=word
        self.activationVector=activationVector
        self.relAct=relAct
        self.relInh=relInh
        self.relevance=relevance
        
    def printMe(self):
        print "Word: "+str(self.index)+" "+self.word
        print "Most Activating Units:"
        print self.relAct
        print "Most Inhibiting Units:"
        print self.relInh
        print "Relevance Vector:"
        print self.relevance

def printListOfList(listA):
    cadena=""
    for y in xrange(len(listA)):
        if y!=0: cadena+=","
        cadena+="("
        for z in xrange(len(listA[y])):
            if z!=0: cadena+=", "
            cadena+=str(listA[y][z])
        cadena+=")"
    return cadena  

'''
Gets a network prodSRNN_notBPTT_mon and plots its weights
'''

def plot_network_weights(srnn):
    monitorW= srnn.W_oh.eval()
    contextW=srnn.W_hh.eval()
    outputW=srnn.W_hy.eval()
    inputW=srnn.W_xh.eval()
    #===========================================================================
    # outputBW=srnn.b.eval()
    # hiddenBW=srnn.bh.eval()
    #===========================================================================
    
    plt.imshow(inputW, cmap='bwr', interpolation='none',vmin=-3.5, vmax=3.5)
    plt.show()
    plt.imshow(monitorW, cmap='bwr', interpolation='none',vmin=-3.5, vmax=3.5)
    plt.show()
    plt.imshow(contextW, cmap='bwr', interpolation='none',vmin=-3.5, vmax=3.5)
    plt.show()
    plt.imshow(outputW, cmap='bwr', interpolation='none',vmin=-4.5, vmax=4.5)
    plt.show()
    
'''
Gets a matrix of weights and plots its histogram
'''   
def plot_weights_histogram(weightMatrix,minV,maxV,binwidth):
    weights=weightMatrix.flatten()
    print "mean:"+str(numpy.mean(weights))
    print "std:"+str(numpy.std(weights))
    
    binsx =numpy.arange(minV,maxV,binwidth)
    plt.hist(weights, bins=binsx)
    plt.show()
    
'''
Gets a network and plots all histograms of weights
'''  
def getAllHistograms(network):
    outputBW=network.b.eval()
    hiddenBW=network.bh.eval()
    
    outputW=network.W_hy.eval()
    monitorW=network.W_oh.eval()
    inputW=network.W_xh.eval()
    contextW=network.W_hh.eval()
    
    print ("output weights")
    plot_weights_histogram(outputW,-3,3.0,0.05)
    print ("monitor weights")
    plot_weights_histogram(monitorW,-3,3.0,0.05)
    print ("input weights")
    plot_weights_histogram(inputW,-3,3.0,0.05)
    print ("context weights")
    plot_weights_histogram(contextW,-3,3.0,0.05)
    
    print ("output bias weights")
    plot_weights_histogram(outputBW,-3,3.0,0.05)
    print ("hidden bias weights")
    plot_weights_histogram(hiddenBW,-3,3.0,0.05)
    
    
def getHiddenActivations(srnn, trainSet):
    activations={}
    counts={}
    h0=srnn.h0
    o0=srnn.o0
    
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
    
    
def getHiddenRelevance(thisSRNN, meanHidActivs,mapIndexWord,normalization=True):
    outputW=thisSRNN.W_hy.eval()
    vocabSize=len(outputW[0])
    
    wordInfos={}
    relevanceHiddenMatrix=[]

    for x in xrange(vocabSize):
        wordWeights=outputW[:,x]
        posW=[w if w>0 else 0 for w in wordWeights]
        negW=[w if w<0 else 0 for w in wordWeights]
        
        actRelevance=meanHidActivs[x]*posW
        inhRelevance=(1-meanHidActivs[x])*negW

        #Normalization
        if normalization:
            zetA=sum(actRelevance)
            zetI=sum(inhRelevance)
            actRelevance=actRelevance/zetA
            inhRelevance=inhRelevance/zetI
             
        #Put together in a single vector
        relevance=actRelevance-inhRelevance
        #===============================================================
        # mostActivs=heapq.nlargest(10, xrange(len(meanHidActivs[x])), meanHidActivs[x].take)
        # leastActivs=heapq.nsmallest(10,xrange(len(meanHidActivs[x])), meanHidActivs[x].take)
        # 
        # mostPosW=heapq.nlargest(10,xrange(len(wordWeights)), wordWeights.take)
        # mostNegW=heapq.nsmallest(10,xrange(len(wordWeights)), wordWeights.take)
        #===============================================================
        
        mostRelAct=heapq.nlargest(5,xrange(len(actRelevance)), actRelevance.take)
        mostRelInh=heapq.nlargest(5,xrange(len(inhRelevance)), inhRelevance.take)
         
        newWord=WordInfoAnalysis(x,mapIndexWord[x],meanHidActivs[x],wordWeights,mostRelAct,mostRelInh, relevance)
        relevanceHiddenMatrix.append(relevance)
        wordInfos[x]= newWord

    return wordInfos,relevanceHiddenMatrix
'''
Gets the dictionary of wordsInfo (returned from getHiddenRelevance) and for each hidden unit, takes the words for which it is most relevant
'''
def getHiddenUnitWords(wordsInfo,noHiddenUnits=120):
    mapAHidWords={}
    mapIHidWords={}
    for x in xrange(noHiddenUnits):
        mapAHidWords[x]=[]
        mapIHidWords[x]=[]
        for y in xrange(43):
            wordI=wordsInfo[y]
            if x in wordI.relAct:
                mapAHidWords[x].append(wordI.index)
            if x in wordI.relInh:
                mapIHidWords[x].append(wordI.index)
    
    return mapAHidWords,mapIHidWords
'''
Gets the dictionary of wordsInfo (returned from getHiddenRelevance) and creates a dictionary pointing from each hidden unit to its relevance values to the output layer
'''
def getRelHidWords(wordsInfo,noHiddenUnits=120):
    mapRelHidWords={}
    for x in xrange(noHiddenUnits):
        mapRelHidWords[x]=[]
        for y in xrange(len(wordsInfo)):
            wordI=wordsInfo[y]
            relWordI=wordI.relevance[x]
            mapRelHidWords[x].append(relWordI)
    return mapRelHidWords
'''
Gets the dictionary of returned from getRelHidWords, and separates positive (activation) from negative values (inhibition), the rest is set to 0
'''
def separateActInhRelHidWords(mapRelHidWords):
    mapActHidWords={}
    mapInhHidWords={}
    for x in xrange(len(mapRelHidWords)):
        rele=mapRelHidWords[x]
        mapActHidWords[x]=[act if act>=0 else 0 for act in rele]
        mapInhHidWords[x]=[act if act<0 else 0 for act in rele]
    return mapActHidWords,mapInhHidWords 

'''
Using Plotly, plot activation and inhibitions for a hidden unit
'''
def plotBarsActInhPerHiddenUnit(acts,inhs,numberUnit,wordList):
    #=======================================================================
    # testUnit=35
    # orderedMapActHidWords=[mapActHidWords[testUnit][x] for x in printList]
    # orderedMapInhHidWords=[mapInhHidWords[testUnit][x] for x in printList]
    # network_analysis.plotBarsActInhPerHiddenUnit(orderedMapActHidWords,orderedMapInhHidWords,testUnit,wordList)
    #=======================================================================
     
    import plotly
    #import plotly.plotly as py    

    trace1 = {
      'x': acts,
      'y': wordList,
      'name': 'Activation',
      'orientation': 'h',
      'type': 'bar',
      'marker': dict(color='rgb(233,91,30)'),
    };
      
    trace2 = {
      'x':  inhs,
      'y': wordList,
      'name': 'Inhibition',
      'orientation': 'h',    
      'type': 'bar',
      'marker': dict(color='rgb(63,184,208)'),
    };

    data = [trace1,trace2];
    layout = {
      'xaxis': {'title': 'Inhibition - Activation'},
      'yaxis': {'title': ''},
      'barmode': 'relative',
      'title': 'Hidden Unit '+str(numberUnit),
      'height':'1000', 
      'width':'400'
    };

    plotly.offline.init_notebook_mode(connected=True) 
    plotly.offline.plot({'data': data, 'layout': layout}, filename='bars-actihn-unit_'+str(numberUnit))
    #py.iplot({'data': data, 'layout': layout}, filename='bars-actihn-unit_'+str(numberUnit))
    
'''
    Get a colormap from matplotlib and convert it to plotly format
'''
def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = map(numpy.uint8, numpy.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale

'''
 Get a name of a colormap and get its corresponding plotly colormap
 ex. name='bwr' or 'seismic'
'''
def getCMapForPlotly(name):
    from matplotlib import cm
    cmap=cm.get_cmap(name)
    return matplotlib_to_plotly(cmap,255)


def createHeatmap(xlabels,ylabels,zvalues,filename,colormap,minV,maxV,title="",height=1000,width=400,offline=True):
    import plotly.graph_objs as go
    import plotly
    
    trace = go.Heatmap(z=zvalues,
                       x=xlabels,
                       y=ylabels,
                       zmin=minV,
                       zmax=maxV,
                       colorscale=colormap
                       )
    data=[trace]
    layout = {
      'xaxis': {'title': '', 'type': 'category'},
      'yaxis': {'title': '', 'type': 'category'},#,'autorange':'reversed'  <-- to flip table vertically
      'title': title,
      'height':height, 
      'width':width
    };
    if offline:
        plotly.offline.init_notebook_mode(connected=True) 
        plotly.offline.plot({'data':data,'layout':layout}, filename=filename)
    else:
        import plotly.plotly as py
        py.iplot({'data':data,'layout':layout}, filename=filename)

def createHeatmapHiddenUnits(relevanceMap, selectedUnits, originalWordsList,printOrderList,filename,colormap,minV,maxV,title="Hidden Units",height=1000,width=400,offline=True):
    import plotly.graph_objs as go
    import plotly
    
    unitValues=[]
    xlabels=[]
    for i in selectedUnits:
        unitValues.append([relevanceMap[i][x] for x in printOrderList])
        xlabels.append(str(i))
    
    #This probably could be done by transposing the list
    zvalues=[]
    for x in xrange(len(originalWordsList)):
        zx=[unit[x] for unit in unitValues]
        zvalues.append(zx)
    
    trace = go.Heatmap(z=zvalues,
                       x=xlabels,
                       y=originalWordsList,
                       zmin=minV,
                       zmax=maxV,
                       colorscale=colormap
                       )
    data=[trace]
    layout = {
      'xaxis': {'title': '', 'type': 'category'},
      'yaxis': {'title': '', 'type': 'category','autorange':'reversed' },# <-- to flip table vertically
      'title': title,
      'height':height, 
      'width':width
    };
    if offline:
        plotly.offline.init_notebook_mode(connected=True) 
        plotly.offline.plot({'data':data,'layout':layout}, filename=filename)
    else:
        import plotly.plotly as py
        py.iplot({'data':data,'layout':layout}, filename=filename)
        

'''
Gets the matrix of relevance of the hidden layer and computes relevance for an input layer
relHiddenMatrix=43x120--->43 words times 120 hidden units
inputMatrix= varx120 ---->var input dimensions times 120 hidden units
output: var x 43
'''
def getRelevanceInput(relHiddenMatrix,inputMatrix,normalization=True):
    relevanceHiddenMatrixTrampose=numpy.matrix.transpose(relHiddenMatrix)
    totalInputRel=inputMatrix*relevanceHiddenMatrixTrampose
    #totalInputRel.shape = var,43
    
    totalInputRelList=totalInputRel.tolist()
    
    activationMatrix=[]
    inhibitionMatrix=[]
    relevanceMatrix=[]
    
    for inputUnit in totalInputRelList:
        rowA=[val if val>=0 else 0 for val in inputUnit]
        rowI=[val if val<0 else 0 for val in inputUnit]
        
        if normalization:
            zI=sum(rowI)
            zA=sum(rowA)
            rowI=[-1.0*val/zI for val in rowI]
            rowA=[val/zA for val in rowA] 
            relInput=[act+inh for act,inh in zip(rowA,rowI)]
        else:
            relInput=inputUnit
        #print relInput
        activationMatrix.append(rowA)
        inhibitionMatrix.append(rowI)
        relevanceMatrix.append(relInput)
        
    return relevanceMatrix,activationMatrix,inhibitionMatrix

def createHeatmapMonitorUnits(monitorMatrix,relevanceHiddenMatrix,colorscale,printList,wordList,normalization=True,filename="monitorHeatmap",minV=-0.3, maxV=0.3,title="Monitoring Units",height=1000,width=900,offline=True):
    import plotly.graph_objs as go

    
    relevanceMatrixMon,actMon,inhMon=getRelevanceInput(relevanceHiddenMatrix,monitorMatrix,normalization)
    
    #ORDER UNITS/DIMENSIONS ACCORDING TO THE ORDER GIVEN IN PRINTLIST
    units=[]
    for unOrderedMon in actMon:
        ordered=[unOrderedMon[x] for x in printList]
        units.append(ordered)     
    unitsPrint=[units[x] for x in printList]
    
    selectedMonWords=range(42) #WE SELECT ALL WORDS EXCEPT THE LAST ONE (PERIOD)
    selWordsValues=[]
    labelsSelWords=[]
    for word in selectedMonWords:
        selWordsValues.append(unitsPrint[word])
        labelsSelWords.append(wordList[word])
 
    #TRANSPOSE
    selWordsMat=numpy.asmatrix(selWordsValues)
    selWordsMatTrans=numpy.matrix.transpose(selWordsMat)
    selWordsValues=selWordsMatTrans.tolist()
 
    trace = go.Heatmap(z=selWordsValues,
                     x=labelsSelWords,
                     y=wordList,
                     zmin=minV,
                     zmax=maxV,
                     colorscale=colorscale
                     )
    data=[trace]
    layout = {
    'xaxis': {'title': ''},
    'yaxis': {'title': '','autorange':'reversed'},
    'title': title,
    'height':height, 
    'width':width
    };
    #py.iplot({'data':data,'layout':layout}, filename='monitorRelevanceAllButPeriod')
    if offline:
        plotly.offline.init_notebook_mode(connected=True) 
        plotly.offline.plot({'data':data,'layout':layout}, filename=filename)
    else:
        import plotly.plotly as py
        py.plot({'data':data,'layout':layout}, filename=filename)
          
def createHeatmapProbs(probsMatrix, colorscale,printList,wordList,normalization=True,filename="probHeatmap",minV=-0.5, maxV=0.5,title="Bigram Probabilities",height=1000,width=900,offline=True):
    import plotly.graph_objs as go
    import plotly
    
    #ORDER UNITS/DIMENSIONS ACCORDING TO THE ORDER GIVEN IN PRINTLIST
    units=[]
    for unOrderedMon in probsMatrix:
        ordered=[unOrderedMon[x] for x in printList]
        units.append(ordered)     
    unitsPrint=[units[x] for x in printList]
    
    selectedMonWords=range(42) #WE SELECT ALL WORDS EXCEPT THE LAST ONE (PERIOD)
    selWordsValues=[]
    labelsSelWords=[]
    for word in selectedMonWords:
        selWordsValues.append(unitsPrint[word])
        labelsSelWords.append(wordList[word])
  
    #TRANSPOSE
    selWordsMat=numpy.asmatrix(selWordsValues)
    selWordsMatTrans=numpy.matrix.transpose(selWordsMat)
    selWordsValues=selWordsMatTrans.tolist()
 
    trace = go.Heatmap(z=selWordsValues,
                     x=labelsSelWords,
                     y=wordList,
                     zmin=minV,
                     zmax=maxV,
                     colorscale=colorscale
                     )
    data=[trace]
    layout = {
    'xaxis': {'title': ''},
    'yaxis': {'title': '','autorange':'reversed'},
    'title': title,
    'height':height, 
    'width':width
    };
    #py.iplot({'data':data,'layout':layout}, filename='monitorRelevanceAllButPeriod')
    if offline:
        plotly.offline.init_notebook_mode(connected=True) 
        plotly.offline.plot({'data':data,'layout':layout}, filename=filename)
    else:
        import plotly.plotly as py
        py.plot({'data':data,'layout':layout}, filename=filename)
                
                

def createHeatmapInputUnits(inputMatrix,relevanceHiddenMatrix,colorscale,printList,wordList,inputLabels,normalization=True,filename="inputHeatmap",minV=-0.3, maxV=0.3,title="Input Units",height=1000,width=900,offline=True):
    import plotly.graph_objs as go
    import plotly
    
    relevanceMatrixInput,actMon,inhMon=getRelevanceInput(relevanceHiddenMatrix,inputMatrix,normalization)
    #relevanceMatrixInput.shape 45x43
    
    #ORDER UNITS/DIMENSIONS ACCORDING TO THE ORDER GIVEN IN PRINTLIST
    units=[]      
    for inputUnit in relevanceMatrixInput:
        units.append([inputUnit[x] for x in printList])
        
    
    #TRANSPOSE
    unitsMatrix=numpy.asmatrix(units)
    unitsMatrixTranspose=numpy.matrix.transpose(unitsMatrix)
    unitsValues=unitsMatrixTranspose.tolist()
 
    trace = go.Heatmap(z=unitsValues,
                       x=inputLabels,
                       y=wordList,
                       zmin=minV,
                       zmax=maxV,
                       colorscale=colorscale
                     )
    data=[trace]
    layout = {
    'xaxis': {'title': ''},
    'yaxis': {'title': '','autorange':'reversed'},
    'title': title,
    'height':height, 
    'width':width
    };
    #py.iplot({'data':data,'layout':layout}, filename='monitorRelevanceAllButPeriod')
    if offline:
        plotly.offline.init_notebook_mode(connected=True) 
        plotly.offline.plot({'data':data,'layout':layout}, filename=filename)
    else:
        import plotly.plotly as py
        py.plot({'data':data,'layout':layout}, filename=filename)            
        

#TIME STEP 0 STUFF
class Weight:
    def __init__(self,value,row,column):
        self.value=value
        self.row=row
        self.column=column
    def printMe(self):
        print [self.value,self.row,self.column]
class WordAct:
    def __init__(self,index,word,activation):
        self.index=index
        self.word=word
        self.activation=activation
    def printMe(self):
        print self.index,self.word,self.activation
         
def dictAddOrAppend(dicty,elemkey,value):
    if dicty.has_key(elemkey):
        dicty[elemkey].append(value)
    else:
        dicty[elemkey]=[value]

#Gets the 10 most positive and negative weights of input->hidden and shows the words related to those hidden units
def getActivationsInhibitionsOf10LargestInputWeights(inputMatrix,mapAHidWords,mapIHidWords,inputUnitsLabels):
    allPosWeights=[]
    allNegWeights=[]
    for x in xrange(45):
        for y in xrange(120):
            weighty=Weight(inputMatrix[x][y],x,y)
            if inputMatrix[x][y]>0:allPosWeights.append(weighty)
            else: allNegWeights.append(weighty)
                 
    sortedPos=sorted(allPosWeights, key=lambda weight: weight.value, reverse=True)
    sortedNeg=sorted(allNegWeights, key=lambda weight: weight.value)

    ##ANALYSIS OF THE 10 HIGHEST WEIGHTS
    sorted10Pos=sortedPos[:10]
    sorted10Neg=sortedNeg[:10]
     
    mapActInputHid={}    
    mapInhInputHid={}
    
    for wei in sorted10Pos:
        if len(mapAHidWords[wei.column])>0:
            dictAddOrAppend(mapActInputHid,wei.row, mapAHidWords[wei.column])
                   
        if len(mapIHidWords[wei.column])>0:
            dictAddOrAppend(mapInhInputHid,wei.row,mapIHidWords[wei.column])

             
    for wei in sorted10Neg:
        if len(mapAHidWords[wei.column])>0:
            dictAddOrAppend(mapInhInputHid, wei.row, mapAHidWords[wei.column])
                    
        if len(mapIHidWords[wei.column])>0:
            dictAddOrAppend(mapActInputHid, wei.row, mapIHidWords[wei.column])

    print mapActInputHid
    print mapInhInputHid
    print "ACTIVATION:"
    for x in xrange(45):
        if mapActInputHid.has_key(x):
            print inputUnitsLabels[x]+" & "+printListOfList(mapActInputHid[x])+"\\\\"
            

    print "INHIBITION:"
    for x in xrange(45):
        if mapInhInputHid.has_key(x):
            print inputUnitsLabels[x]+" & "+printListOfList(mapInhInputHid[x])+"\\\\"    


def separatePositiveNegativeMatrix(originalMatrix):            
    positiveWeightMatrix=[]
    negativeWeightMatrix=[]
    for x in xrange(len(originalMatrix)):
        row=originalMatrix[x]
        rowPos=[val if val>=0 else 0 for val in row]
        rowNeg=[val if val<0 else 0 for val in row]
        
        positiveWeightMatrix.append(rowPos)
        negativeWeightMatrix.append(rowNeg)
    return positiveWeightMatrix,negativeWeightMatrix

def getTotalActivationInhibitionPerWord_OnlyMostPerOutput(inputMatrix,mapActHidWords,mapInhHidWords):
    allPosWeights=[]
    allNegWeights=[]
    #inputW.shape=45x120
    for x in xrange(len(inputMatrix)):
        for y in xrange(len(inputMatrix[0])):
            weighty=Weight(inputMatrix[x][y],x,y)
            if inputMatrix[x][y]>0:allPosWeights.append(weighty)
            else: allNegWeights.append(weighty)
             
    #SUMMING ACTIVATIONS FOR EACH WORD      
    mapWordAct={}
    mapWordInh={}
    for x in xrange(43): #43 is size of vocabulary
        mapWordAct[x]=0
        mapWordInh[x]=0            
                  
    for wei in allPosWeights:
        for wordIndex in mapActHidWords[wei.column]:
            mapWordAct[wordIndex]+=wei.value
                     
        for wordIndex in mapInhHidWords[wei.column]:
            mapWordInh[wordIndex]+=wei.value*-1

    for wei in allNegWeights:
        for wordIndex in mapActHidWords[wei.column]:
            mapWordInh[wordIndex]+=wei.value
            
        for wordIndex in mapInhHidWords[wei.column]:
            mapWordAct[wordIndex]+=wei.value*-1
    
    return mapWordAct,mapWordInh

def sumOutputActivationsInhibitions(outputMatrix,mapIndexWord):        
    activations=[]
    inhibitions=[] 
    #outputMatrix.shape= 120x43
    for x in xrange(outputMatrix.shape[1]):
        wordVector=outputMatrix[:,x]
        wordPos=[val if val>0 else 0 for val in wordVector]
        wordNeg=[val if val<0 else 0 for val in wordVector]    
        wordAct=sum(wordPos)
        wordInh=sum(wordNeg)
        
        activations.append((x,mapIndexWord[x],wordAct))
        inhibitions.append((x,mapIndexWord[x],wordInh))
        
    import operator
    activations.sort(key=operator.itemgetter(2), reverse=True)
    inhibitions.sort(key=operator.itemgetter(2))
    
    for act,inh in zip(activations,inhibitions):
        print act[1] + " & "+str(round(act[2],3)) + " & "+inh[1] + " & " + str(round(inh[2],3))+ "\\\\"

def plotScatter(xvalues,yvalues,filename="scatter-plot",offline=True):
    import plotly
    import plotly.plotly as py
    
    fig, ax = plt.subplots()
    ax.scatter(xvalues,yvalues)
    
    if offline:
        plotly.offline.init_notebook_mode(connected=True)
        plotly.offline.plot_mpl(fig, filename)
    else:
        plot_url = py.plot_mpl(fig, filename)

def createHeatmapContextUnits(contextMatrix,relevanceHiddenMatrix,colorscale,printList,wordList,normalizationCon=True,filename="contextHeatmap",minV=-0.3, maxV=0.3,title="Context Units",height=1000,width=900,offline=True):
    import plotly.graph_objs as go
    import plotly
    import plotly.tools as tools
    from scipy.stats.stats import pearsonr 
    
    relContMatrix,actConMatrix,inhConMatrix=getRelevanceInput(relevanceHiddenMatrix,contextMatrix,normalizationCon)
    
    #Assuming the hidden layer is multiplied with a unit matrix, get its relevance
    unitMatrix=numpy.identity(120)#Just a hack to get hidden relevance transposed and normalized
    relHidMatrix,actHidMatrix,inhHidMatrix=getRelevanceInput(relevanceHiddenMatrix,unitMatrix,normalization=True)

    units=[]

    #INITIALIZE THE LIST WERE THE INFO FOR DIMSCORRS IS TO BE STORED 
    correlsDimsOrig=[]
    correlsDimsCont=[]
    allHidRel=[]
    allContRel=[]
    for k in xrange(len(wordList)):
        correlsDimsOrig.append([])
        correlsDimsCont.append([])
    
    unitsCorrels=[]
    unitsPValues=[]
    
    for i in xrange(len(relHidMatrix)):
        hiddenOutRel=[relHidMatrix[i][x] for x in printList] 
        contextOutRel=[relContMatrix[i][x] for x in printList]
        units.append(hiddenOutRel)
        units.append(contextOutRel)
        
        #PUT VALUES IN LIST TO CALCULATE GLOBAL CORRELATION
        allHidRel=allHidRel+hiddenOutRel
        allContRel=allContRel+contextOutRel
        
        
        #GET CORRELATIONS BETWEEN HIDUNITS AND CONTUNITS
        (withinUnitsCorrs,pvalue)=pearsonr(hiddenOutRel,contextOutRel)
        unitsCorrels.append(withinUnitsCorrs)
        unitsPValues.append(pvalue)
        
        for k in xrange(len(hiddenOutRel)):
            correlsDimsOrig[k].append(hiddenOutRel[k])
            correlsDimsCont[k].append(contextOutRel[k])
            
    #GET CORRELATIONS ACROSS DIMENSIONS
    dimsCorrels=[]
    dimsPValues=[]
    for k in xrange(len(wordList)):
        (corr,pvalue)=pearsonr(correlsDimsOrig[k],correlsDimsCont[k])
        dimsCorrels.append(corr)
        dimsPValues.append(pvalue)

    
    #GLOBAL CORRELATION    
    (corrGlobal,pvalueGlobal)=pearsonr(allHidRel,allContRel)
    print "Global Correlation:"
    print corrGlobal, pvalueGlobal
    print    
    plotScatter(allHidRel,allContRel,"global-scatter")

    #UNITS PLOTS
    #unitIndices=[69,97,9,46]
    unitIndices=[69,46]
    for unitInd in unitIndices:
        unitInd=unitInd*2
        (corr,pvalue)=pearsonr(units[unitInd],units[unitInd+1])
        plotScatter(units[unitInd],units[unitInd+1], "scat_cont_unit_"+str(unitInd))
        print "correlations Unit:"+str(unitInd/2)
        print corr,pvalue
    print
    
    #WORD PLOTS
    #sentenceWordIndices=[13,33,1,0]
    sentenceWordIndices=[13,0]
    for wordInd in sentenceWordIndices:
        plotScatter(correlsDimsOrig[wordInd],correlsDimsCont[wordInd],"scat_cont_word_"+str(wordInd))
        (corr,pvalue)=pearsonr(correlsDimsOrig[wordInd],correlsDimsCont[wordInd])
        print "correlations Word:"+str(wordInd)
        print corr,pvalue
        
    #EXTRACT ONLY THE ONES WE WANT TO PLOT
    numberUnitsToPrint=120
    unitsPrintMatrix=numpy.asmatrix(units[:numberUnitsToPrint*2])
    unitsPrintTrans=numpy.matrix.transpose(unitsPrintMatrix)
    unitsPrint=unitsPrintTrans.tolist()
    
    #GET LABELS OF THE ONES WE WANT TO PLOT
    labelsCon=[]
    for index in xrange(numberUnitsToPrint):
        labelsCon.append(index)
        labelsCon.append(str(index)+"c")
    
    trace2 = go.Heatmap(z=unitsPrint,
                     x=labelsCon[:numberUnitsToPrint*2],
                     y=wordList,
                     zmin=minV,
                     zmax=maxV,
                     colorscale=colorscale,
                     showscale=False
                     )
 
    trace1=go.Heatmap(z=[unitsCorrels[:numberUnitsToPrint]],
                      x=[x for x in range(numberUnitsToPrint)],
                      y=["UnitCorr"],
                      zmin=-1,
                      zmax=1,
                      colorscale=colorscale
                      )
    
    correlsVector=[[corr,0] for corr in dimsCorrels]
    trace3=go.Heatmap(z=correlsVector,
                      x=["DimCorr",""],
                      y=wordList,
                      zmin=-1,
                      zmax=1,
                      colorscale=colorscale
                      )

    fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}],[{'colspan': 2}, None]])

    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace1, 2, 1)

    fig['layout']['xaxis1'].update(title= '', type="category", domain=[0,0.94],side="top")
    fig['layout']['yaxis1'].update(domain=[0.08,1], autorange='reversed')
              
    fig['layout']['xaxis2'].update(title= '',type="category", domain=[0.95,1], ticks='',side="top")
    fig['layout']['yaxis2'].update(type="category",domain=[0.08,1],showticklabels=False, ticks='', autorange='reversed')

    fig['layout']['xaxis3'].update(title='', type="category",domain=[0,0.94],showticklabels=True, ticks='')
    fig['layout']['yaxis3'].update(domain=[0,0.06], ticks='')
    

    if offline:
        plotly.offline.init_notebook_mode(connected=True)
        plotly.offline.plot(fig, filename=filename)
         
    else:
        import plotly.plotly as py
        py.iplot(fig, filename=filename)
    
    
    def getWeakStrongCorrValues(correlations,pvalues,minV,maxV):
        selectedValues=[(corr,pval) for corr,pval in zip(correlations,pvalues) if minV<=corr and corr<maxV]
        pvals=[pval for (corr,pval) in selectedValues]
        if len(pvals)>0:maxPvalue=max(pvals)
        else: maxPvalue=0

        print selectedValues
        print len(selectedValues)
        print maxPvalue
        print
        
        
    #CORRELATIONS OF UNITS    
    getWeakStrongCorrValues(unitsCorrels,unitsPValues,0.6,1.01) #strong correlations
    getWeakStrongCorrValues(unitsCorrels,unitsPValues,0.4,0.6)  #moderate correlations
    getWeakStrongCorrValues(unitsCorrels,unitsPValues,0.2,0.4) #weak correlations
    getWeakStrongCorrValues(unitsCorrels,unitsPValues,-1.0,-0.6) #strong anticorr, none
    getWeakStrongCorrValues(unitsCorrels,unitsPValues,-0.6,-0.4)#moderate anticorr
    getWeakStrongCorrValues(unitsCorrels,unitsPValues,-0.4,-0.2)# weak anticorr
     
    #=======================================================================
    # #CORRELATIONS OF DIMS
    # getWeakStrongCorrValues(correls1,correls2,0.6,1.01) #strong correlations
    # getWeakStrongCorrValues(correls1,correls2,0.4,0.6)  #moderate correlations
    # getWeakStrongCorrValues(correls1,correls2,0.2,0.4) #weak correlations
    # getWeakStrongCorrValues(correls1,correls2,-1.0,-0.6) #strong anticorr, none
    # getWeakStrongCorrValues(correls1,correls2,-0.6,-0.4)#moderate anticorr
    # getWeakStrongCorrValues(correls1,correls2,-0.4,-0.2)# weal anticorr 
    #=======================================================================
    
    
#Used to split a single long string into 2 of maximum length
def splitUnitsString(unitS,length):
    pieces=unitS.split("),")
    cadena1=""
    cadena2=""
    ind=0
    while len(cadena1)+len(pieces[ind])<length:
        if ind!=0:cadena1+=","
        if ind==len(pieces)-1: cadena1+=pieces[ind]
        else: cadena1+=pieces[ind]+")"
        ind+=1
    cadena1+=","
    ind1=ind
    while ind<len(pieces):
        if ind!=ind1:cadena2+=","
        if ind==len(pieces)-1: cadena2+=pieces[ind]
        else: cadena2+=pieces[ind]+")"
        ind+=1
    return cadena1,cadena2


'''
Takes a list of words or vocabulary, and a set of sentences and calculates the bigram probabilities
of that set.
Returns a matrix with such values
'''
def getBigramProbabilities(listOfWords,sentencesSet):
    allCounts={}#counter of bigrams
    for word in listOfWords:
        allCounts[word]={wordV:0 for wordV in listOfWords}
    
    #Count
    for sentence in sentencesSet:
        wordsSentence= sentence.split()
        previousWord=0
        for word in wordsSentence:
            if previousWord:
                allCounts[previousWord][word]+=1
            previousWord=word
    #print allCounts
    
    #Normalize
    for firstWord,secondWords in allCounts.iteritems():
        #print firstWord,secondWords,sum(secondWords.values())
        totalCounts=sum(secondWords.values())*1.0    
        if totalCounts>0:
            allCounts[firstWord]={k: v / totalCounts for k, v in secondWords.iteritems()}    
    
    #Put into a matrix
    matrixProbs=[]
    for word1 in listOfWords:
        rowProb=[allCounts[word1][word2] for word2 in listOfWords]
        matrixProbs.append(rowProb)
    
    return matrixProbs

'''
Rank words according to how likely they occur at the beginning of a sentence
Merges that ranking with the original ranking provided by printList
Returns a list of indices and words corresponding to the new order
'''
def getRankingPos1(sentences,mapWordIndex):
    wordList=[word for word in mapWordIndex.iterkeys()]
    pos1DictCounter={word:0 for word in wordList}
    for sentence in sentences:    
        sentWords=sentence.split()        
        pos1DictCounter[sentWords[0]]+=1
    
    tuplesPos1=pos1DictCounter.items()
            
    import operator
    tuplesPos1.sort(key=operator.itemgetter(1), reverse=True)
    
    rankPos1=[mapWordIndex[word] for (word,_) in tuplesPos1]
    wordListPos1=[word for (word,_) in tuplesPos1]
    
    hideseekIndex=wordListPos1.index("hide_and_seek")
    wordListPos1[hideseekIndex]="hide&seek"
    
            
    def mergeRanks(rankList1,rankList2,mergePoint):
        finalRank=rankList1[:mergePoint]
        for elem in rankList2:
            if elem not in finalRank:
                finalRank.append(elem) 
        return finalRank
        
    rankPos1=mergeRanks(rankPos1,printList,10)
    wordListPos1=[mapIndexWord[i] for i in rankPos1]
    
    return rankPos1,wordListPos1


if __name__ == '__main__':
    import sys
    from data.crossValidation import Fold
    import data.loadFiles as loadFiles
    import rnn.prodSRNN_notBPTT_mon
    
    corpusFilePath="../data/dataFiles/files-thesis/trainTest_Cond-thesis_0.pick"
    modelFilePath="../outputs/prod_main_mon_5cond_outputs/output_beliefVector_120h_0.24lr_200ep_dots_15_40_monitor_sigm_anew/lastModel"
    wordLocalistMapPath='../data/dataFiles/map_localist_words.txt'
    sys.path.append("../data")
   
    s = {
         'nhidden':120,             #number of hidden units
         'seed':345,                #seed for random
         'label':"15_40_monitor_sigm_anew",     #label for this run
         'periods':True,            #whether the corpus has periods
         'inputType':'beliefVector',#dss or sitVector or compVector
         'actpas':True,             #if the inputs are divided in actpas
         'inputFile':corpusFilePath,   #FILE containing the input data
         'modelFile':modelFilePath   #FILE containing the trained model
         }
    if s['periods']: s['vocab_size']=43
    else: s['vocab_size']=42
    
    if s['inputType']=='sitVector' or s['inputType']=='compVector' or s['inputType']=="beliefVector": s['inputDimension']=44
    if s['inputType']=='dss': s['inputDimension']=150
    if s['actpas']:s['inputDimension']=s['inputDimension']+1
    
    
    fold=Fold()
    fold.loadFromPickle(s['inputFile'])
    trainLists=fold.trainSet
    testLists=fold.valtestSet

    loadFiles.setInputType(trainLists[0],s['inputType'])
    for tList in testLists:
        loadFiles.setInputType(tList,s['inputType'])
    
    train=trainLists[0]
    validateList=trainLists[1]
    
    #folderThisRun,bestModel,lastModel,plotsFolder=getFolders(outputsPath,s)
     
    #CREATE SRNN AND LOAD THE PRODUCTION MODEL
    srnn = rnn.prodSRNN_notBPTT_mon.model(
                              inputDimens=s['inputDimension'],
                              hiddenDimens = s['nhidden'],
                              outputDimens= s['vocab_size']
                     )        
    srnn.load(s['modelFile'])
       
    activs,counts=getHiddenActivations(srnn,train)
      
    inputW=srnn.W_xh.eval()
    outputW=srnn.W_hy.eval()
    contextW=srnn.W_hh.eval()
    
    printList=[0,35,9,17,33,7,16,32,10,18,14,31,3,12,22,29,15,37,4,30,6,27,34,20,25,5,21,23,28,39,24,26,41,2,38,11,13,1,8,19,36,40,42]
    
    #Get Vocabulary in the proper order
    mapIndexWord=loadFiles.getWordLocalistMap(wordLocalistMapPath)
    originalWordsList=[word for word in mapIndexWord.itervalues()]#wordList and originalWordsList differ in hide&seek
    mapIndexWord[18]="hide&seek" #instead of hide_and_seek 
    wordList=[mapIndexWord[x] for x in printList]
      
    wordInfos,relevanceHidden=getHiddenRelevance(srnn,activs,mapIndexWord,normalization=True)
    relevanceHiddenMatrix=numpy.asmatrix(relevanceHidden) 
    mapAHidWords,mapIHidWords=getHiddenUnitWords(wordInfos)                
    
    #mapRelHidWords=network_analysis.getRelHidWords(wordInfos) #Get a map from each hidden unit to its relevance values of the output
    #mapActHidWords,mapInhHidWords=network_analysis.separateActInhRelHidWords(mapRelHidWords)#separate the map into activation and inhibition
          
    plotly.tools.set_credentials_file(username='jesusct2', api_key='VoDVZmLfN22kJCln3bCT')
    #plotly.tools.set_credentials_file(username='jesusct', api_key='K0L2vwH3cCZAs1LjdCpQ')        
    
    bwr=getCMapForPlotly("bwr")
      
    #HIDDEN UNITS RELEVANCE!!
    #selectedHUnits=[0,1,2,3,4,10,30,34,35,36,69,80,111,115]
    #network_analysis.createHeatmapHiddenUnits(mapRelHidWords,selectedHUnits,wordList,printList,filename="selectedHUnits",colormap=bwr,minV=-0.11,maxV=0.11,title="Selected Hidden Units1")
    
    #MONITORING RELEVANCE!!!!
    #monitorW=srnn.W_oh.eval()
    #createHeatmapMonitorUnits(monitorW,relevanceHiddenMatrix,bwr,printList,wordList,True,filename="monitorHeatmapAct",title="Monitoring Units Activation",height=1000,width=900,offline=False)
    
    
    #===========================================================================
    # #GET BIGRAM PROBABILITIES HEATMAP      
    # trainingSents=[item.testItem for item in trainLists[0]]              
    # matrixProbs=getBigramProbabilities(originalWordsList, trainingSents)
    # createHeatmapProbs(matrixProbs,bwr,printList,wordList,True,filename="probsHeatmap1",title="Monitoring Probs1",height=1000,width=900,offline=False)
    #===========================================================================

  
    #INPUUUUUT STUFF RELEVANCE!!!!  
    inputUnitsLabels=["play(charlie,chess)","play(charlie,hide&seek)","play(charlie,soccer)","play(heidi,chess)","play(heidi,hide&seek)","play(heidi,soccer)",
                 "play(sophia,chess)","play(sophia,hide&seek)","play(sophia,soccer)","play(charlie,puzzle)","play(charlie,ball)","play(charlie,doll)",
                 "play(heidi,puzzle)","play(heidi,ball)","play(heidi,doll)","play(sophia,puzzle)","play(sophia,ball)","play(sophia,doll)",
                 "win(charlie)","win(heidi)","win(sophia)","lose(charlie)","lose(heidi)","lose(sophia)","place(charlie,bathroom)","place(charlie,bedroom)",
                 "place(charlie,playground)","place(charlie,street)","place(heidi,bathroom)","place(heidi,bedroom)","place(heidi,playground)","place(heidi,street)",
                 "place(sophia,bathroom)","place(sophia,bedroom)","place(sophia,playground)","place(sophia,street)","manner(play(charlie),well)","manner(play(charlie),badly)",
                 "manner(play(heidi),well)","manner(play(heidi),badly)","manner(play(sophia),well)","manner(play(sophia),badly)","manner(win,easily)","manner(win,difficultly)","actives"]
    #===========================================================================
    #  
    # createHeatmapInputUnits(inputW,relevanceHiddenMatrix,bwr,printList,wordList,inputUnitsLabels,normalization=False,filename="testinput",minV=-2,maxV=2,title="Input Units",height=1200,width=1000,offline=True)
    #        
    # #CONTEXT RELEVANCE!!!!
    # createHeatmapContextUnits(contextW,relevanceHiddenMatrix,bwr,printList,wordList,normalizationCon=True,filename="contextHeatmapALL",minV=-0.4, maxV=0.4,title="Context Units",height=1000,width=900,offline=True)
    # 
    # #TIME STEP 0
    # #Gets the 10 most positive and negative weights of input->hidden and shows the words related to those hidden units
    # getActivationsInhibitionsOf10LargestInputWeights(inputW,mapAHidWords,mapIHidWords,inputUnitsLabels)
    # 
    # posM,negM=separatePositiveNegativeMatrix(inputW)
    # mapA,mapI=getTotalActivationInhibitionPerWord_OnlyMostPerOutput(inputW,mapAHidWords,mapIHidWords)
    # sumOutputActivationsInhibitions(outputW,mapIndexWord)
    #===========================================================================
    
    sents = [item.testItem for item in train]
    mapIndexWord[18]='hide_and_seek'
    mapWordIndex={word:index for index,word in mapIndexWord.iteritems()}
    
    rankPos1,wordListPos1=getRankingPos1(sents, mapWordIndex)
    #original Ranking
    #createHeatmapInputUnits(inputW,relevanceHiddenMatrix,bwr,printList,wordList,inputUnitsLabels,normalization=False,filename="non-normalizedInputRelevance",minV=-2,maxV=2,title="Input Units Non-normalized",height=1200,width=1000,offline=False)    
    #first the ones possible at t0
    createHeatmapInputUnits(inputW,relevanceHiddenMatrix,bwr,rankPos1,wordListPos1,inputUnitsLabels,normalization=False,filename="rankPos1T0test",minV=-2,maxV=2,title="Input Units RankPos1 T0test",height=1200,width=1000,offline=False)
   

