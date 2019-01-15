import theano,numpy,os
from theano import tensor as T
from collections import OrderedDict

'''
Recurrent NeuralNetwork of a language model. Receives one token and the hidden layer at t=t-1 and gives a distribution over the next words
It assumes the first word it receives is NOT the sentence boundary or <s> (it is added automatically as w0)
'''

class model(object):   
    def __init__(self, inputDimens,hiddenDimens,outputDimens,startSentIndex):
      
        def sample_weights(sizeX, sizeY):
            values = numpy.ndarray([sizeX, sizeY], dtype=theano.config.floatX)  #@UndefinedVariable
            for dx in xrange(sizeX):
                vals=numpy.random.normal(loc=0.0, scale=0.1,size=(sizeY,))
                values[dx,:] = vals
            return values

        # parameters of the model
        self.W_wh   = theano.shared(sample_weights(inputDimens,hiddenDimens))
        self.W_hh   = theano.shared(sample_weights(hiddenDimens,hiddenDimens))
        self.W_hy   = theano.shared(sample_weights(hiddenDimens,outputDimens))
        
        self.bh  = theano.shared(numpy.zeros(hiddenDimens, dtype=theano.config.floatX)) #@UndefinedVariable
        self.b   = theano.shared(numpy.zeros(outputDimens, dtype=theano.config.floatX)) #@UndefinedVariable
        
        #fixed constants
        self.h0  = numpy.zeros(hiddenDimens, dtype=theano.config.floatX)  # @UndefinedVariable
        self.w0  = numpy.zeros(outputDimens, dtype=theano.config.floatX)  # @UndefinedVariable 
        if startSentIndex>-1: #if we want all sentences to start by something, otherwise just empty
            self.w0[startSentIndex]=1
       
        
        # bundle
        self.params = [self.W_wh, self.W_hh, self.W_hy, self.bh, self.b]
        self.names  = ['W_wh', 'W_hh', 'W_hy', 'bh', 'b']
        wordIn = T.vector("wordIn") #   
        wordOut   = T.vector("wordOut") #words in localist representation
        h_tm1 = T.vector("h_tm1")
        
        h_t = T.tanh(T.dot(wordIn, self.W_wh) + T.dot(h_tm1, self.W_hh) + self.bh)
        predWordLoc = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b)

        #wordPred = T.argmax(predWordLoc,axis=1) #index of word with highest activation
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')
        loss = -T.mean(wordOut * T.log(predWordLoc) + (1.- wordOut) * T.log(1. - predWordLoc)) #Cross entropy 
        #loss=T.nnet.categorical_crossentropy(predWordLoc,wordOut)
        
 
        gradients = T.grad(loss, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[wordIn,wordOut,h_tm1], outputs=[predWordLoc,h_t,loss])

        self.train = theano.function( inputs  = [wordIn, wordOut, lr,h_tm1],
                                      outputs = [loss,h_t],
                                      updates = updates )
        
     
    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
    
    def load(self, folder):
        for param, name in zip(self.params, self.names):
            values =numpy.load(os.path.join(folder, name + '.npy'))
            param.set_value(values)
            
    '''
    Takes the randomized training set (a set of sentences in localist form) 
    and trains an epoch
    returns a list with error values for each 25th training item
    Each error value is the cross entropy between expected and obtained
    '''    
    def epochTrain(self,trainSetLocalist,learningRate):
        errors=[]
        for sentIndex in xrange(len(trainSetLocalist)):
            sentence=trainSetLocalist[sentIndex]            
            
            wordIn=self.w0
            h_tm1=self.h0
            errSent=0
    
            for  wordOut in sentence:
                [e,h_tm1]=self.train(wordIn,wordOut,learningRate,h_tm1)
                wordIn=wordOut
                errSent+=e
                     
            if sentIndex%25==0:
                errors.append(errSent) 
        return errors
    
    '''
    Takes the validation set (a set of sentences in localist form) and obtains
    the predictions of the model, without modifying connection weights
    Each error value is the cross entropy between expected and obtained
    '''   
    def epochValidate(self, valSetLocalist):
        errors=[]
    
        for sentIndex in xrange(len(valSetLocalist)):
            sentence=valSetLocalist[sentIndex]            
    
            wordIn=self.w0
            h_tm1=self.h0
            errSent=0
            for wordOut in sentence:
                [_,h_tm1,loss]=self.classify(wordIn,wordOut,h_tm1)
                wordIn=wordOut
                errSent+=loss
                     
            if sentIndex%25==0:
                errors.append(errSent) 
        return errors
            
            
    '''
    Takes a sentence, in the form of wordsLocalist and words, and returns triplets (word,index,probability) 
    and together with the sentence probability
    '''   
    def getSentenceWordProbs(self,wordsLocalist,sentenceWords):
        sentenceWordIndices=[numpy.argmax(localist) for localist in wordsLocalist]
        sentenceProbs=[]
        
        word0=self.w0
        h_tm1=self.h0
        sentP=1.0
        
        for wordLoc,wordIndex,sentenceWord in zip(wordsLocalist,sentenceWordIndices,sentenceWords):
            [outProbs,h_tm1,_]=self.classify(word0,wordLoc,h_tm1)
            
            word0=wordLoc
            wordP=outProbs[0][wordIndex]
            sentenceProbs.append((sentenceWord,wordIndex,wordP))
            
            sentP=sentP*wordP
        return sentP,sentenceProbs 

    
    
