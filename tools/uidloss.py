import numpy

def UIDLoss(probsSeq):
    mean=0
    for prob in probsSeq:
        mean+=prob
    mean=mean*1.0/len(probsSeq)
    print mean
    sd=0
    for prob in probsSeq:
        sd+=(prob-mean)*(prob-mean)
        
    sd=sd*1.0/len(probsSeq)

    return numpy.sqrt(sd)

probs=[0.4,0.5,0.5,0.6,0.5,0.51]
print UIDLoss(probs)


'''
Defines a ranking loss
'''
def rankingLoss(perfect,proposed):
    weIndex=0
    suma=0
    for (elemPer,elemProp) in zip (perfect,proposed):
        newElem=(elemPer-elemProp)*1.0/(weIndex+1)
        weIndex+=1
        sum+=newElem
    return suma