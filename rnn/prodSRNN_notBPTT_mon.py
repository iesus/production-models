import theano,numpy,os
from theano import tensor as T
from collections import OrderedDict


'''
    Recurrent Neural Network to model production, receives a DSS input and outputs a word per time step
    it learns with backpropagation. One could also use backpropagation through time, but per Matt's request, we don't.
    However, there is no apparent difference in performance, it only takes more time to train if one uses backpropagation through time.
'''

class model(object):   
    def __init__(self, inputDimens,hiddenDimens,outputDimens):
        
        #rescale such that the largest singular value = 1, not really used
        def rescale_weights(values):
            _,svs,_ =numpy.linalg.svd(values)
            values = values /svs[0] #svs[0] is the largest singular value 
            return values
        
        def sample_weights(sizeX, sizeY):
            values = numpy.ndarray([sizeX, sizeY], dtype=theano.config.floatX)  # @UndefinedVariable
            for dx in xrange(sizeX):
                vals=numpy.random.normal(loc=0.0, scale=0.1,size=(sizeY,))
                values[dx,:] = vals
            #rescale_weights(values)
            return values

        # parameters of the model
        self.W_xh   = theano.shared(sample_weights(inputDimens,hiddenDimens))
        self.W_oh   = theano.shared(sample_weights(outputDimens,hiddenDimens))
        self.W_hh   = theano.shared(sample_weights(hiddenDimens,hiddenDimens))
        self.W_hy   = theano.shared(sample_weights(hiddenDimens,outputDimens))
        
        self.bh  = theano.shared(numpy.zeros(hiddenDimens, dtype=theano.config.floatX))  # @UndefinedVariable
        self.b   = theano.shared(numpy.zeros(outputDimens, dtype=theano.config.floatX))  # @UndefinedVariable
        
        #fixed constants
        self.h0  = numpy.zeros(hiddenDimens, dtype=theano.config.floatX)  # @UndefinedVariable
        self.o0  = numpy.zeros(outputDimens, dtype=theano.config.floatX)  # @UndefinedVariable

        # bundle
        self.params = [self.W_xh,self.W_oh, self.W_hh, self.W_hy, self.bh, self.b]
        self.names  = ['W_xh','W_oh','W_hh', 'W_hy', 'bh', 'b']
        dss = T.vector("dss") 
        wordLoc   = T.vector("y") #words in localist representation
        h_tm1 = T.vector("h_tm1")
        o_tm1 = T.vector("o_tm1")
        
        h_t = T.nnet.sigmoid(T.dot(dss, self.W_xh) + T.dot(h_tm1, self.W_hh) + T.dot(o_tm1,self.W_oh)+ self.bh)
        
        outputWordProbs = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b)
        wordPredLoc = T.argmax(outputWordProbs,axis=1)
        
        # loss, gradients and learning rate
        lr = T.scalar('lr')
        loss = -T.mean(wordLoc * T.log(outputWordProbs) + (1.- wordLoc) * T.log(1. - outputWordProbs)) #Cross entropy loss
         
        gradients = T.grad(loss, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[dss,h_tm1,o_tm1], outputs=[wordPredLoc,h_t,outputWordProbs[0]]) 
        #outputWordProbs at this time is the future o_tm1

        self.train = theano.function( inputs  = [dss, wordLoc, lr,h_tm1,o_tm1],
                                      outputs = [loss,h_t,outputWordProbs[0]],
                                      updates = updates )
     
    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
    
    def load(self, folder):
        for param, name in zip(self.params, self.names):
            values =numpy.load(os.path.join(folder, name + '.npy'))
            param.set_value(values)
            
    
    def epochTrain(self,trainSet,learningRate):
        '''
        Takes the randomized training set (a set of TrainingElement) and trains an epoch
        returns a list with error values for each 25th training item
        '''
        errors=[]
        for sentIndex in xrange(len(trainSet)):
            sentence=trainSet[sentIndex]            
            
            h_tm1=self.h0 
            o_tm1=self.o0
            errSent=0
            for word in sentence.wordsLocalist:
                [e,h_tm1,_]=self.train(sentence.input,word,learningRate,h_tm1,o_tm1)
                o_tm1=word
    
                errSent+=e
                     
            if sentIndex%25==0:
                errors.append(errSent) 
        return errors
            
    
    def getSentenceProb(self,semantics,wordsLocalist):
        '''
        Takes a semantic representation, and a sentence in localist form, and calculates its conditional probability
        P(sent|DSS)
        '''
        sentenceWordIndices=[numpy.argmax(localist) for localist in wordsLocalist]
        wordInLoc=self.o0
        h_tm1=self.h0
        sentP=1.0
        
        for wordOutLoc,wordIndex in zip(wordsLocalist,sentenceWordIndices):
            [_,h_tm1,outProbs]=self.classify(semantics,h_tm1,wordInLoc)
            wordInLoc=wordOutLoc
            wordP=outProbs[wordIndex]
            sentP=sentP*wordP
            print wordP
    
        return sentP  
    

    def getModelProductions(self,testSet,periods=True):
        '''
        Takes a testSet (a list of TrainingElement) and whether the predictions should stop by a period or by the expected sentence length 
        Returns the word indices of the sentences produced by the model
        '''
        productions=[] 
        
        for item in testSet:
            sentenceProduced=[]
            h_tm1=self.h0 
            o_tm1=self.o0
            predWord=0
         
            if periods:
                while predWord<42 and len(sentenceProduced)<20:
                    [predWord,h_tm1,o_tm1]=self.classify(item.input,h_tm1,o_tm1)
                    predWord=predWord[0]
                    o_tm1=self.o0.copy()
                    o_tm1[predWord]=1.0
                    
                    sentenceProduced.append(predWord)
            else: 
                for _ in xrange(len(item.wordsLocalist)):
                    [predWord,h_tm1,o_tm1]=self.classify(item.input,h_tm1,o_tm1)
                    predWord =predWord[0]
                    o_tm1=self.o0.copy()
                    o_tm1[predWord]=1.0
                    
                    sentenceProduced.append(predWord)
                
            productions.append(sentenceProduced)
        return productions

        
