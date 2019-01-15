import theano,numpy,os
from theano import tensor as T
from collections import OrderedDict

import rnn.prodSRNN_notBPTT_mon

'''
    RecurentNeuralNetwork to model UID, receives a probability distribution of possible word productions and a DSS, and outputs a new probability distribution per time step that hopefully follows UID
    Essentially, it reranks the distribution that is given as input.
'''

class model(object):   
    def __init__(self,productionModelPath, vocabSize,hiddenDimensProd,hiddenDimensRerank,dssDimens):
        
        self.loadProductionModel(productionModelPath, dssDimens, hiddenDimensProd, vocabSize)
        
        #PARAMETERS OUTPUT COMBINATION LAYER
        self.probSlope=-1
        self.probIntercept=3.5
        self.derLenSlope=1
        self.derLenIntercept=-2.5
        
        
        def sample_weights(sizeX, sizeY):
            values = numpy.ndarray([sizeX, sizeY], dtype=theano.config.floatX)  # @UndefinedVariable
            for dx in xrange(sizeX):
                vals=numpy.random.normal(loc=0.0, scale=0.1,size=(sizeY,))
                values[dx,:] = vals
            return values

        # parameters of the model
        self.W_wh   = theano.shared(sample_weights(vocabSize,hiddenDimensRerank))
        self.W_dssh   = theano.shared(sample_weights(dssDimens,hiddenDimensRerank))
        self.W_hh   = theano.shared(sample_weights(hiddenDimensRerank,hiddenDimensRerank))
        self.W_hy   = theano.shared(sample_weights(hiddenDimensRerank,vocabSize))
        
        self.bh  = theano.shared(numpy.zeros(hiddenDimensRerank, dtype=theano.config.floatX))  # @UndefinedVariable
        self.b   = theano.shared(numpy.zeros(vocabSize, dtype=theano.config.floatX))  # @UndefinedVariable
        
        self.h0R  = numpy.zeros(hiddenDimensRerank, dtype=theano.config.floatX)  # @UndefinedVariable
        self.o0  = numpy.zeros(vocabSize, dtype=theano.config.floatX)  # @UndefinedVariable
        
        # bundle
        self.params = [self.W_wh,self.W_dssh, self.W_hh, self.W_hy, self.bh, self.b]
        self.names  = ['W_wh','W_dssh','W_hh', 'W_hy', 'bh', 'b']
        softIn = T.vector("softIn") #   
        softOut   = T.vector("y") #words in localist representation
        h_tm1 = T.vector("h_tm1")
        dss = T.vector("dss")

        h_t = T.nnet.sigmoid(T.dot(softIn, self.W_wh) + T.dot(h_tm1, self.W_hh) + T.dot(dss,self.W_dssh)+ self.bh)
        predOutDistro = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        loss = -T.mean(softOut * T.log(predOutDistro) + (1.- softOut) * T.log(1. - predOutDistro)) #Cross entropy 
        #loss=T.nnet.categorical_crossentropy(predOutDistro,softOut)
 
        gradients = T.grad(loss, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[dss,softIn,h_tm1], outputs=[h_t,predOutDistro[0]]) #predOutDistro at this time is the future o_tm1

        self.train = theano.function( inputs  = [dss,softIn, softOut, lr,h_tm1],
                                      outputs = [loss,h_t,predOutDistro[0]],
                                      updates = updates )
        
     
    def loadProductionModel(self,productionModelPath,dssDimens,hiddenDimens,outputDimens):
        simpleProdModel= rnn.prodSRNN_notBPTT_mon.model(
                              inputDimens=dssDimens,
                              hiddenDimens = hiddenDimens,
                              outputDimens= outputDimens
                              )  
        simpleProdModel.load(productionModelPath)
        self.simpleProdModel=simpleProdModel
        
        
    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
    
    def load(self, folder):
        for param, name in zip(self.params, self.names):
            values =numpy.load(os.path.join(folder, name + '.npy'))
            param.set_value(values)
        
        
    def epochTrain(self,trainSet,learningRate):
        errors=[]
        for sentIndex in xrange(len(trainSet)):
            item=trainSet[sentIndex]
            expectedLengthVectors=numpy.asarray(item.lengthVectors)

            h_tm1LM=self.simpleProdModel.h0
            o_tm1=self.o0
            h_tm1RR=self.h0R
            errSent=0
            
            for i in xrange(len(item.wordsLocalist)):
                [_,h_tm1LM,o_tm1]=self.simpleProdModel.classify(item.input,h_tm1LM,o_tm1)
                [e,h_tm1RR,_]=self.train(item.input,o_tm1,expectedLengthVectors[i],learningRate,h_tm1RR)
                o_tm1=item.wordsLocalist[i]
    
                errSent+=e
                     
            if sentIndex%25==0:
                errors.append(errSent) 
        return errors
    
    def getModelPredictions(self,testSet,periods,combine=False):
        predictions_test=[] 
        
        for sentIndex in xrange(len(testSet)):
            sentence=testSet[sentIndex]            
            predSent=[]
       
            h_tm1=self.simpleProdModel.h0
            o_tm1=self.o0
            h_tm1R=self.h0R
            indexW=0
            
            if periods:
                while indexW<42 and len(predSent)<20:
                    [_,h_tm1,o_tm1]=self.simpleProdModel.classify(sentence.input,h_tm1,o_tm1)
                    [h_tm1R,predOutDistro]=self.classify(sentence.input,o_tm1,h_tm1R)
                    
                    if combine:score=self.combineScores(predOutDistro, o_tm1, sentence)
                    else: score=predOutDistro

                    indexW=numpy.argmax(score)
                    o_tm1=self.o0.copy()
                    o_tm1[indexW]=1.0
                    
                    predSent.append(indexW)
            else: 
                for _ in xrange(len(sentence.wordsLocalist)):
                    [_,h_tm1,o_tm1]=self.simpleProdModel.classify(sentence.input,h_tm1,o_tm1)
                    [h_tm1R,predOutDistro]=self.classify(sentence.input,o_tm1,h_tm1R)

                    if combine:score=self.combineScores(predOutDistro, o_tm1, sentence)
                    else: score=predOutDistro

                    indexW = numpy.argmax(score)
                    o_tm1=self.o0.copy()
                    o_tm1[indexW]=1.0
                    
                    predSent.append(indexW)
                
            predictions_test.append(predSent)
        return predictions_test
    
    def combineScores(self,lengthScores,probScores,trainingIt):
        p_dss=len(trainingIt.equivalents)*1.0/130
        
        probMultiplier=self.probSlope*p_dss+self.probIntercept
        derLenMultiplier=-self.derLenSlope*p_dss+self.derLenIntercept
        
        combination=derLenMultiplier*lengthScores+probScores*probMultiplier
        return combination
            
        
        
        
        

